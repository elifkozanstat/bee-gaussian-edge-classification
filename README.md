# bee-gaussian-edge-classification# Gaussian–Edge Feature Pipelines for Bee Genus Classification  
**Public code and materials accompanying the manuscript**

This repository provides the full implementation, preprocessing modules, and analysis utilities used to evaluate Gaussian-filtered grayscale images combined with classical edge operators (Sobel, Prewitt, Laplacian of Gaussian, Canny) for genus-level bee classification (Apis vs. Bombus).  
The materials hosted here support the reproducibility and transparency of the statistical analyses described in the associated manuscript.

The repository includes:

- Image preprocessing and feature construction pipeline  
- Edge-based feature extraction (Sobel, Prewitt, LoG, Canny)  
- Classical classifiers (Logistic Regression, SVM, Random Forest, Gradient Boosting)  
- Imbalance-handling strategies (no adjustment, class weights, SMOTE)  
- Scripts to reproduce all experimental results  
- Scripts to generate the figures used in the manuscript  
- A reference results file (`results_all_from_tables.csv`) containing all reported metrics  

---

## Repository Structure

bee-gaussian-edge-classification/
├─ src/
│ └─ bee_gaussian_pipeline/
│ ├─ data.py # Image loading & preprocessing
│ ├─ features.py # Filters & edge operators
│ ├─ models.py # Statistical classifiers
│ └─ evaluate.py # Train/test evaluation routines
├─ scripts/
│ ├─ run_experiments.py # Runs the full factorial experiment
│ └─ plot_results.py # Generates the manuscript figures
├─ results/
│ └─ results_all_from_tables.csv # Exact results used in manuscript
├─ notebooks/
│ └─ 01_experiments_colab.ipynb # Optional development notebook
├─ requirements.txt
└─ README.md
