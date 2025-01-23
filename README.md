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
The project workflow is structured as follows:

1. **Exploratory Data Analysis (EDA)**:
   - Investigated data distributions, relationships, and potential issues.
   - Identified class imbalances and applied SMOTE to address them.

2. **Feature Engineering**:
   - Scaled numerical features for improved model performance.
   - Created new features to enhance predictive power.

3. **Baseline Model Training**:
   - Trained multiple machine learning models using default parameters.
   - Evaluated models using Accuracy, F1-Score, and ROC-AUC metrics.

4. **Hyperparameter Tuning**:
   - Optimized high-performing models such as Random Forest and Gradient Boosting.
   - Used GridSearchCV for systematic hyperparameter exploration.

5. **Model Comparison**:
   - Compared tuned models to identify the best performer.
   - Gradient Boosting achieved the highest ROC-AUC score (0.9994).

6. **Final Evaluation and Deployment Preparation**:
   - Conducted final model validation on unseen data.
   - Saved the best-performing model as a `.pkl` file for deployment.

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
git clone <repository_url>

# Step 2: Navigate to the project directory
cd obesity-level-prediction

# Step 3: Install the required libraries
pip install -r requirements.txt

# Step 4: Open the Jupyter Notebook
jupyter notebook Obesity_Level_Estimation_Final.ipynb

# Step 5: Run the notebook cells sequentially to reproduce the analysis and results

## Future Work
While the model performed well, future improvements could include:
1. Expanding the dataset to include more diverse geographic regions.
2. Testing additional machine learning models such as AutoML solutions.
3. Exploring deep learning techniques for enhanced performance.

---

## Acknowledgments
- **Dataset**: The dataset was sourced from a publicly available repository.
- **Tools**: Analysis was performed using Python and libraries such as Pandas, Scikit-learn, and Matplotlib.
- **Inspiration**: This project was inspired by the global effort to combat obesity and improve public health.

---

## Contact
For questions or collaboration, please reach out at:
- **Email**: [your_email@example.com]
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
