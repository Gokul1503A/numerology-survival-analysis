# Numerology and Survival Rate Analysis

This project explores the relationship between **numerology-based features** (derived from names) and survival rates using machine learning (ML) and deep learning (DL) models. The project leverages the Titanic dataset as a case study and investigates whether numerological patterns influence survival predictions.

## Project Overview

The project includes the following steps:
1. **Feature Engineering**: Deriving numerology-based features from passenger names.
   - Name Numerology
   - Soul Number
   - Personality Number
   - Name Length
2. **Data Preparation**: Cleaning and splitting the Titanic dataset for ML/DL models.
3. **Model Training**:
   - **Machine Learning**: Logistic Regression and Random Forest
   - **Deep Learning**: Fully connected neural networks using PyTorch.
4. **Model Evaluation**: Visualizing confusion matrices, accuracy, precision, and recall.


## How to Run
Follow these steps to run the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gokul1503A/numerology-survival-analysis.git
   cd numerology-survival-analysis
   ```

2. Set up the environment:
   ```bash
   conda create -n numerology_env python=3.9
   conda activate numerology_env
   pip install -r requirements.txt
   conda install --file conda-requirements.txt
   ```

3. Run the scripts:
   ```bash
   python ml_model.py
   python dl_model.py
   ```

4. View the results:
   - Check the confusion matrix and metrics in the pictures below.
   <img src="/ml_confusion_matrices.png" alt="ML Confusion Matrix" width="500">
   <img src="/dl_confusion_matrix.png" alt="DL Confusion Matrix" width="500">

## Results
The results include confusion matrices, accuracy, precision, and recall values for:
- Logistic Regression
- Random Forest
- Deep Learning Neural Network

Visualizations demonstrate model performance and comparison between ML and DL approaches.

## Future Work
- Explore other survival-based datasets.
- Add additional numerology features for enhanced predictions.


## Acknowledgments
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- Libraries: PyTorch, Scikit-Learn, Pandas, and Matplotlib

---

Thank you for exploring this project! 🚀
