# Thyroid Disease Prediction using SVM, Decision Tree, and K-Nearest Neighbors Models

This repository contains a Python script for predicting thyroid disease using three different machine learning models: Support Vector Machine (SVM), Decision Tree Classifier (DTC), and K-Nearest Neighbors (KNN). The purpose of this project is to compare the performance of these three models and determine which one achieves the maximum accuracy in predicting thyroid disease.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Comparison](#model-comparison)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)

## Introduction

Thyroid disease is a common endocrine disorder that affects the thyroid gland's function. Early detection and accurate prediction of thyroid disease are crucial for effective medical treatment. Machine learning models can be used to predict the likelihood of thyroid disease based on patient data.

In this project, we have implemented three machine learning models: SVM, DTC, and KNN to predict thyroid disease. We will compare the performance of these models to determine which one provides the highest accuracy in disease prediction.

## Dataset

We have used a dataset containing various features related to thyroid disease, including patient demographics, medical history, and test results. The dataset has been preprocessed and split into training and testing sets for model evaluation.

## Installation

To run the prediction script, you need to have Python installed on your system. Additionally, you will need the following libraries:

- scikit-learn
- pandas
- numpy

You can install these libraries using pip:

```bash
pip install scikit-learn pandas numpy
```

## Usage

To use the provided code for thyroid disease prediction, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/Arun-AK03/Thyroid-Disease-Prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd thyroid-disease-prediction
   ```

3. Run the prediction script:

   ```bash
   python thyroid_prediction.py
   ```

This script will train the SVM, DTC, and KNN models on the training data and evaluate their performance on the testing data.

## Model Comparison

We have implemented three different models for thyroid disease prediction: SVM, DTC, and KNN. After running the script, the accuracy of each model will be displayed, and the model with the highest accuracy will be identified.

## Results

The script will display the accuracy of each model on the testing data. Here are the results:

- Support Vector Machine (SVM) Accuracy: 94%
- Decision Tree Classifier (DTC) Accuracy: 99%
- K-Nearest Neighbors (KNN) Accuracy: 97%

The model with the highest accuracy is the best choice for predicting thyroid disease in this dataset.

## Conclusion

In this project, we have compared three machine learning models for predicting thyroid disease: SVM, DTC, and KNN. The accuracy results from testing data help identify the most accurate model for this specific dataset. Accurate prediction of thyroid disease is vital for early diagnosis and treatment.

## Contributing

If you would like to contribute to this project or improve the models, feel free to open an issue or submit a pull request.
