# Sex Classification Using Machine Learning

This project leverages machine learning to classify biological sex based on facial features extracted from images. Using datasets such as the CFD (Chicago Face Database) and FRLLS, this project explores multiple stages of data processing, feature engineering, and model development.

---

## Objectives
- **Data Preparation**: Process datasets by extracting relevant features (e.g., 68-point facial landmarks).
- **Feature Engineering**: Calculate meaningful ratios and distances from facial landmarks to create a robust feature set.
- **Classification**: Train machine learning models to classify sex with high accuracy.
- **Evaluation**: Assess model performance on validation and independent test datasets.

---

## Methodology
1. **Data Preprocessing**:
    - Facial landmark extraction using Dlib's 68-point model.
    - Aspect ratio adjustments to ensure uniform input image dimensions.
    - Outlier handling and feature standardization for consistency.

2. **Feature Engineering**:
    - Calculate distances, symmetry ratios, and proportions from facial landmarks.
    - Create derived features such as "golden ratios" and inter-ocular distances.

3. **Exploratory Data Analysis**:
    - Feature distribution visualization using KDE plots and box plots.
    - Pairwise correlation heatmaps to identify feature relationships.
    - Variance Inflation Factor (VIF) analysis to mitigate multicollinearity.

4. **Model Training**:
    - Evaluate classical models (Logistic Regression, Random Forest, SVM, Gradient Boosting) and advanced techniques (Neural Networks).
    - Use feature selection (e.g., Recursive Feature Elimination) to identify the most predictive variables.

5. **Performance Metrics**:
    - Assess models on metrics such as accuracy, precision, recall, and F1-score.
    - Employ cross-validation for model stability evaluation.

6. **Generalization**:
    - Test models on an independent dataset (FRLLS) to evaluate real-world applicability.

---

## Tools and Technologies
- **Libraries**: Python with OpenCV, Dlib, Scikit-learn, TensorFlow, Matplotlib, Seaborn.
- **Datasets**: CFD (Chicago Face Database) and FRLLS.
- **Frameworks**: Keras for Neural Networks, Statsmodels for statistical analysis.

---

## Required Libraries
To run this project, the following Python libraries must be installed:
- **Computer Vision:**
  - OpenCV-Python
  - Dlib
- **Data Analysis and Machine Learning:**
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow
  - Statsmodels
- **Data Visualization:**
  - Matplotlib
  - Seaborn
- **Progress Bars:**
  - tqdm
 
---

## Results
- Achieved high classification accuracy (>90%) on validation datasets.
- Identified the top predictive features, reducing model complexity without sacrificing performance.
- Demonstrated generalizability through testing on independent datasets.

---

## Future Work
- Extend the feature set to include texture and color-based metrics.
- Investigate the use of deep learning models for end-to-end feature extraction and classification.
- Incorporate diverse datasets to address potential biases.

---

---

## Acknowledgments
- **Dlib Library**: For robust facial landmark detection.
- **CFD and FRLLS Datasets**: Providing high-quality facial images for analysis.
- **Scikit-learn and TensorFlow**: For machine learning implementations.

