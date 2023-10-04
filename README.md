# Customer_Churn

The primary goal of this project is to train a decision tree classifier on customer data and assess its accuracy in predicting customer churn. It follows common machine learning practices for data preprocessing, model training, and evaluation.

**#This Python code performs the following tasks:**

**Import Libraries: The code begins by importing the necessary libraries:**

pandas (as pd) for data manipulation and handling DataFrames.
DecisionTreeClassifier from scikit-learn (sklearn.tree) for creating and training a decision tree classifier.
train_test_split from sklearn.model_selection for splitting the dataset into training and testing sets.
accuracy_score from sklearn.metrics for evaluating the accuracy of the model's predictions.
Read Customer Data: It reads customer data from a CSV file named 'Customer_data.csv' into a Pandas DataFrame called customer_data.

**Data Preparation:**

It selects specific columns ('age', 'monthly_payment', 'contract_length') from the DataFrame as the feature variables (X).
It selects the 'churned' column as the target variable (Y).
Splitting Data: It splits the dataset into training and testing sets using train_test_split. The training set (X_train and Y_train) is used to train the model, and the testing set (X_test and Y_test) is used to evaluate its performance. The testing set comprises 10% of the entire dataset (test_size=0.1).

**Model Creation and Training:**

It creates an instance of the DecisionTreeClassifier as model.
It fits (trains) the decision tree model using the training data (X_train and Y_train).

**Making Predictions:**

It uses the trained model to make predictions on the test data (X_test), which is stored in sample_to_predict.
The predictions are stored in the predictions variable.

**Evaluating Model Accuracy:**

It calculates the accuracy score of the model's predictions using the true labels (Y_test) and the predicted labels (predictions). The accuracy score measures the proportion of correctly classified samples in the test set.
The accuracy score is stored in the score variable.
Printing Accuracy Score: It prints the accuracy score to the console, indicating how well the decision tree model performed on the test data.
