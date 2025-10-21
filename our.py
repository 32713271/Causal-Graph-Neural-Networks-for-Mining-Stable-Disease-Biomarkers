# improved_gcn_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, mean_squared_error, confusion_matrix, 
                           f1_score, roc_curve, auc, roc_auc_score)
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from scipy.stats import pearsonr
from scipy.sparse import lil_matrix, coo_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import warnings
warnings.filterwarnings("ignore")

class DataLoader:
    @staticmethod
    def data_get():
        disese = pd.read_csv(r"D:\jupyter\New\dateset\mRNA_LncRNA_Non_Small_Cell_Lung_Carcinoma.csv")
        health = pd.read_csv(r"D:\jupyter\New\dateset\mRNA_LncRNA_Non_Small_Cell_Lung_Carcinoma_HC.csv")
        
        dat = disese[disese['gene_type'] == "protein_coding"]
        tes = health[health['gene_type'] == "protein_coding"]
        column_name = dat.iloc[:, 2].values
        dat_samples = dat.iloc[:, 6:].T
        tes_samples = tes.iloc[:, 6:].T
        
        return dat_samples, tes_samples, column_name

    @staticmethod
    def prepare_data():
        dat_ace, tes_ace, column_name = DataLoader.data_get()
        df_data = pd.concat([dat_ace, tes_ace], axis=0)
        df_data.columns = [column_name]
        
        dis_lab = [1] * len(dat_ace)
        HC_lab = [0] * len(tes_ace)
        label = dis_lab + HC_lab
        
        df_Z = scale(df_data, axis=0, with_mean=True, with_std=True, copy=True)
        expr_data = pd.DataFrame(df_Z, columns=df_data.columns.get_level_values(0).values)
        
        return expr_data, label

class ImprovedGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(ImprovedGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_node_features)
        
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        identity = x
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        
        return x + identity

class ImprovedGraphLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(ImprovedGraphLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, predictions, targets, edge_index):
        mse_loss = F.mse_loss(predictions, targets)
        row, col = edge_index
        edge_loss = torch.mean(torch.abs(
            torch.norm(predictions[row] - predictions[col], dim=1) -
            torch.norm(targets[row] - targets[col], dim=1)
        ))
        return mse_loss + self.alpha * edge_loss

class GCNTrainer:
    def __init__(self, expr_data, interaction_data_path):
        self.expr_data = expr_data
        self.interaction_data_path = interaction_data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_graph_data(self):
        interaction_data = pd.read_csv(self.interaction_data_path, sep="\t", header=None)
        interaction_data.columns = ['raid', 'actor1', 'cat1', 'sp1', 'actor2', 'cat2', 'sp2', 'score']
        
        genes = self.expr_data.columns
        adj_matrix = lil_matrix((len(genes), len(genes)))
        gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        
        for index, row in interaction_data.iterrows():
            if row['actor1'] in gene_to_idx and row['actor2'] in gene_to_idx:
                i, j = gene_to_idx[row['actor1']], gene_to_idx[row['actor2']]
                adj_matrix[i, j] = adj_matrix[j, i] = 1
        
        adj_matrix_csr = adj_matrix.tocsr()
        adj_matrix_coo = coo_matrix(adj_matrix_csr)
        edge_index = torch.tensor([adj_matrix_coo.row, adj_matrix_coo.col], dtype=torch.long)
        x = torch.tensor(self.expr_data.values.T, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index)

    def train_model(self, data, epochs=500):
        input_dim = data.x.shape[1]
        model = ImprovedGCN(input_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        criterion = ImprovedGraphLoss()
        
        best_loss = float('inf')
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            out = model(data)
            loss = criterion(out, data.x, data.edge_index)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)
            
            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(model)
            
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
        return best_model

class FeatureAnalyzer:
    @staticmethod
    def calculate_pearson_correlations(data_gcn, expr_data):
        data_gcn = data_gcn.loc[:, ~data_gcn.columns.duplicated(keep='first')]
        expr_data = expr_data.loc[:, ~expr_data.columns.duplicated(keep='first')]
        
        pearson_scores = []
        data_gcn = data_gcn.fillna(0)
        expr_data = expr_data.fillna(0)
        
        for gene in data_gcn.columns:
            try:
                gcn_values = pd.to_numeric(data_gcn[gene].squeeze(), errors='coerce')
                original_values = pd.to_numeric(expr_data[gene].squeeze(), errors='coerce')
                
                valid_data = pd.DataFrame({
                    'gcn': gcn_values,
                    'original': original_values
                }).dropna()
                
                if not valid_data.empty:
                    correlation, _ = pearsonr(valid_data['gcn'], valid_data['original'])
                    pearson_scores.append({"Gene": gene, "Pearson_Correlation": correlation})
            except Exception as e:
                print(f"Error processing gene {gene}: {e}")
                continue
        
        return pd.DataFrame(pearson_scores)

class ModelEvaluator:
    def __init__(self):
        self.classifiers = {
            'SVC': SVC(probability=True),
            'XGBClassifier': XGBClassifier(),
            'GaussianNB': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier()
        }

    @staticmethod
    def plot_confusion_matrices(y_true, all_predictions, filename):
        plt.figure(figsize=(20, 15))
        classes = np.unique(y_true)
        
        for idx, (clf_name, predictions) in enumerate(all_predictions.items(), 1):
            plt.subplot(2, 2, idx)
            cm = confusion_matrix(y_true, predictions, labels=classes)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=classes, 
                        yticklabels=classes)
            plt.title(f"{clf_name} Confusion Matrix")
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    @staticmethod
    def plot_roc_curves(y_true, classifiers_probas, filename):
        plt.figure(figsize=(10, 8))
        
        for clf_name, probas in classifiers_probas.items():
            if probas is not None:
                fpr, tpr, _ = roc_curve(y_true, probas)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{clf_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    # 数据加载和预处理
    expr_data, label = DataLoader.prepare_data()
    
    # GCN模型训练
    trainer = GCNTrainer(expr_data, r'D:\jupyter\new_funtion\2024_7_9\download.txt')
    graph_data = trainer.prepare_graph_data()
    best_model = trainer.train_model(graph_data)
    
    # 特征分析
    out = best_model(graph_data)
    out_np = out.detach().cpu().numpy()
    data_gcn = pd.DataFrame(out_np.T, columns=expr_data.columns, index=expr_data.index)
    
    # 计算相关性
    pearson_df = FeatureAnalyzer.calculate_pearson_correlations(data_gcn, expr_data)
    filtered_genes = pearson_df[
        (pearson_df["Pearson_Correlation"] >= 0.7) | 
        (pearson_df["Pearson_Correlation"] <= -0.7)
    ]
    
    # 模型评估
    evaluator = ModelEvaluator()
    feature_ranges = [50, 100, 200]
    
    for max_features in feature_ranges:
        evaluator.evaluate_features(data_gcn, label, filtered_genes["Gene"], max_features)

if __name__ == "__main__":
    main()