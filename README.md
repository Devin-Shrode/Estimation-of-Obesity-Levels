{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1e766b22-0dea-42f2-82a6-ee10255f1db5",
   "metadata": {},
   "source": [
    "# Obesity Level Prediction: A Machine Learning Approach\n",
    "\n",
    "This project focuses on predicting obesity levels (`NObeyesdad`) based on physical and behavioral attributes. The analysis spans data exploration, feature engineering, and machine learning model development. The project leverages a variety of models, selects the best-performing one, and prepares it for deployment.\n",
    "\n",
    "---\n",
    "\n",
    "## Table of Contents\n",
    "1. [Introduction](#introduction)\n",
    "2. [Dataset Description](#dataset-description)\n",
    "3. [Workflow](#workflow)\n",
    "4. [Results](#results)\n",
    "5. [How to Run the Project](#how-to-run-the-project)\n",
    "6. [Future Work](#future-work)\n",
    "7. [Acknowledgments](#acknowledgments)\n",
    "8. [Contact](#contact)\n",
    "\n",
    "---\n",
    "\n",
    "## Introduction\n",
    "Obesity is a global health concern, associated with numerous medical conditions such as diabetes, cardiovascular diseases, and certain types of cancer. This project aims to develop a machine learning solution for accurately predicting obesity levels using data related to eating habits, physical activity, and other behavioral factors.\n",
    "\n",
    "---\n",
    "\n",
    "## Dataset Description\n",
    "The dataset used in this project contains records from Mexico, Peru, and Colombia. It includes attributes that detail physical and behavioral characteristics, such as:\n",
    "- Physical attributes: Age, Height, Weight.\n",
    "- Behavioral attributes: Daily calorie intake, exercise frequency, water consumption.\n",
    "- Target variable: `NObeyesdad` (Obesity Level) with seven possible categories.\n",
    "\n",
    "**Key Statistics**:\n",
    "- **Total records**: 2,111\n",
    "- **Number of features**: 17 (including the target variable)\n",
    "- **Class distribution**:\n",
    "  - Insufficient Weight\n",
    "  - Normal Weight\n",
    "  - Overweight Level I and II\n",
    "  - Obesity Type I, II, and III\n",
    "\n",
    "---\n",
    "\n",
    "## Workflow\n",
    "The project workflow is structured as follows:\n",
    "\n",
    "1. **Exploratory Data Analysis (EDA)**:\n",
    "   - Investigated data distributions, relationships, and potential issues.\n",
    "   - Identified class imbalances and applied SMOTE to address them.\n",
    "\n",
    "2. **Feature Engineering**:\n",
    "   - Scaled numerical features for improved model performance.\n",
    "   - Created new features to enhance predictive power.\n",
    "\n",
    "3. **Baseline Model Training**:\n",
    "   - Trained multiple machine learning models using default parameters.\n",
    "   - Evaluated models using Accuracy, F1-Score, and ROC-AUC metrics.\n",
    "\n",
    "4. **Hyperparameter Tuning**:\n",
    "   - Optimized high-performing models such as Random Forest and Gradient Boosting.\n",
    "   - Used GridSearchCV for systematic hyperparameter exploration.\n",
    "\n",
    "5. **Model Comparison**:\n",
    "   - Compared tuned models to identify the best performer.\n",
    "   - Gradient Boosting achieved the highest ROC-AUC score (0.9994).\n",
    "\n",
    "6. **Final Evaluation and Deployment Preparation**:\n",
    "   - Conducted final model validation on unseen data.\n",
    "   - Saved the best-performing model as a `.pkl` file for deployment.\n",
    "\n",
    "---\n",
    "\n",
    "## Results\n",
    "The Gradient Boosting model was selected as the final model due to its:\n",
    "- Superior ROC-AUC (0.9994), indicating strong classification performance.\n",
    "- Consistent accuracy and F1-score across all obesity levels.\n",
    "\n",
    "Random Forest was recommended as an alternative for scenarios prioritizing interpretability.\n",
    "\n",
    "**Feature Importance**:\n",
    "The top features contributing to the predictions were:\n",
    "1. Weight\n",
    "2. Height\n",
    "3. Daily vegetable consumption (FCVC)\n",
    "\n",
    "---\n",
    "\n",
    "## How to Run the Project\n",
    "Follow these steps to reproduce the analysis and results:\n",
    "\n",
    "```bash\n",
    "# Step 1: Clone the repository\n",
    "git clone <repository_url>\n",
    "\n",
    "# Step 2: Navigate to the project directory\n",
    "cd obesity-level-prediction\n",
    "\n",
    "# Step 3: Install the required libraries\n",
    "pip install -r requirements.txt\n",
    "\n",
    "# Step 4: Open the Jupyter Notebook\n",
    "jupyter notebook Obesity_Level_Estimation_Final.ipynb\n",
    "\n",
    "# Step 5: Run the notebook cells sequentially to reproduce the analysis and results\n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "## Future Work\r\n",
    "While the model performed well, future improvements could include:\r\n",
    "1. Expanding the dataset to include more diverse geographic regions.\r\n",
    "2. Testing additional machine learning models such as AutoML solutions.\r\n",
    "3. Exploring deep learning techniques for enhanced performance.\r\n",
    "\r\n",
    "---\r\n",
    "\r\n",
    "## Acknowledgments\r\n",
    "- **Dataset**: The dataset was sourced from a publicly available repository.\r\n",
    "- **Tools**: Analysis was performed using Python and libraries such as Pandas, Scikit-learn, and Matplotlib.\r\n",
    "- **Inspiration**: This project was inspired by the global effort to combat obesity and improve public health.\r\n",
    "\r\n",
    "---\r\n",
    "\r\n",
    "## Contact\r\n",
    "For questions or collaboration, please reach out at:\r\n",
    "- **Email**: [your_email@example.com]\r\n",
    "- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)\r\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "921bac85-5731-4086-81f8-fd6af5cd2390",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "93356f84-289a-42dd-b3e7-4efe1ac441d4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
