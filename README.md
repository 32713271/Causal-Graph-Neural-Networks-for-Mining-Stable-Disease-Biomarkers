Causal Graph Neural Networks for Mining Stable Disease Biomarkers
Overview

This repository contains the code and data for the paper "Causal Graph Neural Networks for Mining Stable Disease Biomarkers". The work focuses on biomarker discovery from high-throughput transcriptomic data using causal inference and graph neural networks (GNNs). Our method identifies stable biomarkers by considering gene-gene regulatory relationships, improving upon traditional approaches that may conflate spurious correlations with genuine causal effects.

The proposed method integrates causal inference with multi-layer graph neural networks (GNNs). The key innovation of this work is the inclusion of a causal effect estimation process to ensure that biomarkers are stable and reproducible across multiple datasets.

Dataset

The datasets used in this study come from multiple sources:

PltDB: This database includes RNA-seq data for diseases such as breast cancer, non-small cell lung cancer (NSCLC), and glioblastoma.

GEO (Gene Expression Omnibus): Datasets such as GSE33000 and GSE44770 were used for Alzheimer’s Disease (AD) biomarker discovery.

Data processing steps include:

Extracting mRNA expression from blood RNA-seq samples.

Removing low-quality or redundant entries.

Handling missing values with mean imputation.

Standardizing gene expression data to a uniform scale.

Methodology

The proposed method consists of the following steps:

Gene Regulatory Network Construction: Building a gene regulatory graph where nodes represent genes and edges indicate co-expression relationships.

Propensity Scoring Using GNN: Using a three-layer GNN to calculate propensity scores based on co-regulated genes, capturing indirect regulatory effects.

Causal Effect Estimation: Estimating the causal effect of each gene on the disease phenotype using logistic regression.

Comparisons with Other Methods

The effectiveness of our method was compared with the following baseline algorithms:

CFS-master: Feature selection in the contrastive analysis setting.

MCFS: Unsupervised feature selection for multi-cluster data.

Fastcan: Orthogonal least squares-based fast feature selection for linear classification.

DFS: Deep feature screening for ultra-high-dimensional data via deep neural networks.

Traditional Causal Inference: Standard causal inference method without the integration of GNN for causal structure learning.
