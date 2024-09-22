import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import pickle

# Load datasets
train_dataset = pd.read_csv('train_dataset.csv')
test_dataset = pd.read_csv('test_dataset.csv')

# Preprocess data
train_dataset['type'] = train_dataset['type'].map({'e': 1, 'p': 0})
test_dataset['type'] = test_dataset['type'].map({'e': 1, 'p': 0})

train_dataset = pd.get_dummies(train_dataset, dtype=int)
test_dataset = pd.get_dummies(test_dataset, dtype=int)

# Prepare training data
x_train = train_dataset.drop(['type','cap_shape_c'], axis=1)
y_train = train_dataset["type"]

# Prepare test data
x_test = test_dataset.drop("type", axis=1)
y_test = test_dataset["type"]

for col in x_train.columns:
    if col not in x_test.columns:
        x_test[col] = 0
x_test = x_test[x_train.columns]

# Train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(x_train, y_train)

# Make predictions
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

# Calculate accuracies
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f"Test Accuracy: {accuracy_test}, Train Accuracy: {accuracy_train}")

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)