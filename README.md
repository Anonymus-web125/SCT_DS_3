# SCT_DS_3
## Decision Tree Classifier for Predicting Customer Purchases
### Overview
This repository contains a project from my Data Science internship at SkillCraft Technology, where I built a decision tree classifier to predict whether a customer will purchase a product or service. The project uses the Bank Marketing dataset from the UCI Machine Learning Repository, which contains demographic and behavioral data on customers.

### Contents
Jupyter Notebook: The main analysis and model-building process is contained in a Jupyter notebook (SCT_DS_3.ipynb).
Dataset: The Bank Marketing dataset (bank-full.csv) is included in the repository.
### Project Structure
SCT_DS_3.ipynb: The notebook contains the complete code for data preprocessing, model building, and evaluation.
bank-full.csv: The dataset used for training and testing the decision tree classifier.
### Data Preprocessing
The dataset was preprocessed to ensure it was ready for modeling. Key preprocessing steps included:

Handling Missing Values: Dealing with any missing data to ensure a complete dataset.
Encoding Categorical Variables: Converting categorical features into numerical values using techniques like one-hot encoding.
Feature Selection: Identifying and selecting relevant features to improve model performance.
### Model Building
The decision tree classifier was built using the following steps:

Splitting the Data: The dataset was split into training and testing sets to evaluate model performance.
Training the Model: A decision tree classifier was trained on the training data.
Hyperparameter Tuning: Various hyperparameters were adjusted to optimize the model’s performance.
### Model Evaluation
The model was evaluated using several metrics to assess its performance, including:

Accuracy: The overall correctness of the model's predictions.
Confusion Matrix: To visualize the performance of the classifier.
Precision, Recall, and F1-Score: To provide deeper insights into the model's ability to correctly classify customers.
### Key Findings
The decision tree classifier provided valuable insights into the factors that influence whether a customer will purchase a product or service. The results highlight the importance of certain demographic and behavioral features in predicting customer behavior.

### How to Use
Clone this repository to your local machine.
Ensure you have the necessary Python packages installed (e.g., pandas, scikit-learn, matplotlib).
Open the SCT_DS_3.ipynb notebook in Jupyter and run the cells to replicate the analysis.
### Conclusion
This project demonstrates the utility of decision tree classifiers in predicting customer behavior based on demographic and behavioral data.
### Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request.

### Contact
For any questions or feedback, feel free to contact me at bikashkumargiri23@gmail.com.
