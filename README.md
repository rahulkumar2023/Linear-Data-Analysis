# Linear Data Analysis Repsoitory

This repository contains implementations and analysis of various machine learning and data analysis techniques using MATLAB. Detailed explanations of the results and analysis are included in the accompanying PDF files for each script.

## Files and Descriptions

### 1. **binary_classification_pca.m** & **binary_classification_pca.pdf**
- **Purpose**: Implements binary classification using Principal Component Analysis (PCA). The script evaluates models like logistic regression and artificial neural networks (ANN) to classify data effectively.
- **Inputs**:
  - `collegenum.csv`: Dataset used for PCA and classification.
- **Outputs**: Results include accuracy, AUC scores, and visualizations of separating hyperplanes and ROC curves.
- **Details**: The accompanying PDF file, `binary_classification_pca.pdf`, contains explanations of the methodology, results, and analysis.

---

### 2. **diabetes_obesity_analysis.m** & **diabetes_obesity_analysis.pdf**
- **Purpose**: Analysis of diabetes and obesity-related data to uncover relationships and predictive insights.
- **Inputs**:
  - `dmrisk.csv`: Dataset containing diabetes and obesity metrics.
- **Outputs**: Insights and visualizations related to health indicators.
- **Details**: All explanations and analysis are included in `diabetes_obesity_analysis.pdf`.

---

### 3. **fragility_analysis.m** & **fragility_analysis.pdf**
- **Purpose**: Examines fragility metrics to evaluate health-related trends, with a focus on male demographic data.
- **Inputs**:
  - `fragility2013male.csv`: Dataset used for the analysis.
- **Outputs**: Plots and statistical results.
- **Details**: The PDF file, `fragility_analysis.pdf`, provides the complete analysis and interpretations.

---

### 4. **pca_clustering_analysis.m** & **pca_clustering_analysis.pdf**
- **Purpose**: Explores PCA for dimensionality reduction and clustering techniques to analyze datasets.
- **Inputs**:
  - `wine.csv`: Dataset used for clustering analysis.
- **Outputs**: Clustering visualizations and insights into data groupings.
- **Details**: Detailed explanations and clustering analysis are provided in `pca_clustering_analysis.pdf`.

---

### 5. **spectral_clustering.m** & **spectral_clustering.pdf**
- **Purpose**: Implements spectral clustering for graph-based data analysis and clustering.
- **Inputs**:
  - `graph_edges.txt`: Edge list representing graph connections for clustering.
- **Outputs**: Spectral clustering results and cluster visualizations.
- **Details**: The accompanying PDF file, `spectral_clustering.pdf`, contains detailed analysis and explanations.

---

## Datasets

### 1. **collegenum.csv**
- Data for PCA and binary classification in `binary_classification_pca.m`.

### 2. **dmrisk.csv**
- Contains metrics for diabetes and obesity analysis in `diabetes_obesity_analysis.m`.

### 3. **fragility2013male.csv**
- Dataset for analyzing fragility trends in `fragility_analysis.m`.

### 4. **wine.csv**
- Dataset used for PCA and clustering in `pca_clustering_analysis.m`.

### 5. **graph_edges.txt**
- Edge list for graph-based spectral clustering in `spectral_clustering.m`.

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repo-name
   ```
3. Run MATLAB scripts:
   - Open `.m` files in MATLAB and execute them to generate results.
   - Ensure input datasets are in the same directory.
4. Refer to the corresponding PDF files for detailed analysis and explanations of the results.

---
