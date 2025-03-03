# Obesity Level Prediction: A Machine Learning Approach

This project focuses on predicting obesity levels (`NObeyesdad`) based on physical and behavioral attributes. The analysis spans data exploration, feature engineering, and machine learning model development. The project leverages a variety of models, selects the best-performing one, and prepares it for deployment.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Workflow](#workflow)
4. [Results](#results)
5. [How to Run the Project](#how-to-run-the-project)
6. [Future Work](#future-work)
7. [Acknowledgments](#acknowledgments)
8. [Contact](#contact)

---

## Introduction
Obesity is a global health concern, associated with numerous medical conditions such as diabetes, cardiovascular diseases, and certain types of cancer. This project aims to develop a machine learning solution for accurately predicting obesity levels using data related to eating habits, physical activity, and other behavioral factors.

---

## Dataset Description
The dataset used in this project contains records from Mexico, Peru, and Colombia. It includes attributes that detail physical and behavioral characteristics, such as:
- Physical attributes: Age, Height, Weight.
- Behavioral attributes: Daily calorie intake, exercise frequency, water consumption.
- Target variable: `NObeyesdad` (Obesity Level) with seven possible categories.

**Key Statistics**:
- **Total records**: 2,111
- **Number of features**: 17 (including the target variable)
- **Class distribution**:
  - Insufficient Weight
  - Normal Weight
  - Overweight Level I and II
  - Obesity Type I, II, and III

---

## Workflow

1. **Problem Understanding**:
   - The goal was to predict obesity levels based on eating habits and physical condition using a multi-class classification approach.

2. **Exploratory Data Analysis (EDA)**:
   - Conducted comprehensive EDA to understand feature distributions, relationships, and class imbalances.
   - Visualized key trends and patterns to guide preprocessing and modeling decisions.

3. **Data Preprocessing**:
   - Cleaned the dataset by handling missing values and encoding categorical variables.
   - Scaled numerical features to ensure uniformity across different scales.
   - Addressed class imbalance using oversampling techniques.

4. **Model Training and Evaluation**:
   - Trained multiple machine learning models, including Logistic Regression, Random Forest, Gradient Boosting, Support Vector Classifier, Multi-Layer Perceptron, and others.
   - Evaluated models using Accuracy, F1-Score, and ROC-AUC metrics.
   - Conducted hyperparameter tuning for key models to optimize their performance.

5. **Model Comparison**:
   - Compared the performance of tuned models to select the best one for deployment.

6. **Feature Importance Analysis**:
   - Analyzed feature importance for the best-performing model to provide insights into key predictors.

7. **Final Evaluation**:
   - Validated the best model on unseen test data to confirm robustness and generalizability.

8. **Deployment Preparation**:
   - Saved the final model as a pickle file for easy deployment.
   - Documented the workflow and results for reproducibility.

---

## Results
The Gradient Boosting model was selected as the final model due to its:
- Superior ROC-AUC (0.9994), indicating strong classification performance.
- Consistent accuracy and F1-score across all obesity levels.

Random Forest was recommended as an alternative for scenarios prioritizing interpretability.

**Feature Importance**:
The top features contributing to the predictions were:
1. Weight
2. Height
3. Daily vegetable consumption (FCVC)

---

## How to Run the Project
Follow these steps to reproduce the analysis and results:

```bash
# Step 1: Clone the repository
git clone <https://github.com/Devin-Shrode/Estimation-of-Obesity-Levels>

# Step 2: Navigate to the project directory
cd Estimation-of-Obesity-Levels

# Step 3: Install the required libraries
pip install -r requirements.txt

# Step 4: Open the Jupyter Notebook
jupyter notebook Obesity_Level_Estimation_Final.ipynb

# Step 5: Run the notebook cells sequentially to reproduce the analysis and results

```

---

## Future Work
While the model performed well, future improvements could include:
1. Expanding the dataset to include more diverse geographic regions.
2. Testing additional machine learning models such as AutoML solutions.
3. Exploring deep learning techniques for enhanced performance.

---

## Acknowledgments
- **Dataset**: The dataset was sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).
- **Tools**: Analysis was performed using Python and libraries such as Pandas, Scikit-learn, and Matplotlib.
- **Inspiration**: This project was inspired by the global effort to combat obesity and improve public health.

---

## Contact
For any questions or collaboration opportunities, reach out at:
- **Email**: devin.shrode@proton.me  
- **LinkedIn**: [linkedin.com/in/DevinShrode](https://www.linkedin.com/in/DevinShrode)  
- **GitHub**: [github.com/Devin-Shrode/Wine-Quality](https://github.com/Devin-Shrode)  

