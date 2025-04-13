# Breast Cancer Classification Models

## Overview
This repository contains a collection of logistic regression models trained on the **Breast Cancer Wisconsin dataset** for binary classification of tumors (malignant vs. benign). The models were developed using **scikit-learn** as part of a machine learning assignment to explore different optimization techniques and regularization methods.

## Models
The following models and preprocessing objects are included in this repository:

- **scaler.joblib**: A `StandardScaler` used for preprocessing features (required for non-pipeline models).
- **batch_model.joblib**: Logistic Regression trained with Batch Gradient Descent (using the `lbfgs` solver).
- **sgd_model.joblib**: Logistic Regression trained with Stochastic Gradient Descent (SGD).
- **mini_batch_model.joblib**: Logistic Regression trained with Mini-batch Gradient Descent (approximated using SGD).
- **poly_pipeline.joblib**: A pipeline combining `PolynomialFeatures` (degree=2), `StandardScaler`, and Logistic Regression.
- **l2_model.joblib**: Logistic Regression with L2 (Ridge) regularization.
- **es_model.joblib**: Logistic Regression with Early Stopping.

## Dataset
- **Source**: Breast Cancer Wisconsin dataset, accessed via `sklearn.datasets.load_breast_cancer`.
- **Features**: 30 numerical features (e.g., mean radius, mean texture, mean perimeter).
- **Target**: Binary classification (0 = malignant, 1 = benign).
- **Size**: 569 samples.
- **Split**: 80% training (455 samples), 20% validation (114 samples).
- **Preprocessing**: Features were standardized using `StandardScaler` (except for the `poly_pipeline`, which handles scaling internally).

## Training Details
- **Library**: `scikit-learn`.
  
- **Optimization Techniques**:
  - Batch Gradient Descent: Used `lbfgs` solver with `max_iter=100`.
  - Stochastic Gradient Descent: Used `SGDClassifier` with `loss='log_loss'`, constant learning rate (`eta0=0.01`), and `max_iter=100`.
  - Mini-batch Gradient Descent: Approximated using `SGDClassifier` with shuffling enabled.
  - Polynomial Features: Added degree-2 polynomial features, followed by scaling and logistic regression.
  - L2 Regularization: Applied with `C=1.0` and `max_iter=1000`.
  - Early Stopping: Used `SGDClassifier` with `early_stopping=True`, validation fraction of 0.1, and `n_iter_no_change=10`.
    
- **Random State**: Set to 42 for reproducibility across all models.

## Evaluation Metrics
The models were evaluated on the validation set (114 samples) using accuracy and confusion matrices. Below are the accuracy scores:

| Model                  | Accuracy  |
|-----------------------|-----------|
| Batch GD              | 97.37%    |
| SGD                   | 98.25%    |
| Mini-batch GD         | 98.25%    |
| Polynomial GD         | 97.37%    |
| Early Stopping        | 99.12%    |

Confusion matrices for each model are available in the [original notebook](#) (link to notebook can be added if shared).

## Usage
### Installation
Ensure you have `scikit-learn` and `joblib` installed:
```python
pip install scikit-learn joblib
```
**Loading and Using Non-Pipeline Models**

For models like batch_model, sgd_model, mini_batch_model, l2_model, and es_model, you need the scaler for preprocessing:
```python
import joblib
import numpy as np

# Load the scaler and model
scaler = joblib.load('scaler.joblib')
model = joblib.load('batch_model.joblib')  # Replace with desired model

# Example: Preprocess new data (replace with your data)
X_new = np.array([[17.99, 10.38, 122.80, ...]])  # 30 features
X_new_scaled = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_new_scaled)
print(predictions)  # 0 (malignant) or 1 (benign)
```
**Loading and Using the Pipeline Model**
- The poly_pipeline includes its own preprocessing steps, so the scaler is not needed:

```python
import joblib
import numpy as np

# Load the pipeline
poly_pipeline = joblib.load('poly_pipeline.joblib')

# Example: New data (replace with your data)
X_new = np.array([[17.99, 10.38, 122.80, ...]])  # 30 features

# Make predictions directly
predictions = poly_pipeline.predict(X_new)
print(predictions)  # 0 (malignant) or 1 (benign)
```
## Intended Use
- These models are intended for educational purposes, demonstrating the application of logistic regression with various optimization techniques on a medical dataset. They can be used for:
 - Classifying breast tumors as malignant or benign based on 30 features.
 - Comparing the performance of different gradient descent methods and regularization techniques.

## Limitations
- **Dataset Size:** The dataset is relatively small (569 samples), which may limit model generalization.
- **Feature Engineering:** Only polynomial features (degree=2) were explored; other feature engineering techniques might improve performance.
- **Model Complexity:** Logistic regression is a linear model and may not capture complex patterns as well as non-linear models (e.g., SVM, neural networks).
- **Evaluation:** Performance was evaluated on a single validation split; cross-validation could provide a more robust assessment.

## License
- This project is licensed under the MIT License.

## Author
- Created by Arsal Mairaj on April 11, 2025.

For questions or contributions, please open an issue in the repository.
