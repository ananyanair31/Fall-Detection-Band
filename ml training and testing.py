import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob

# Step 1: Read the dataset
directory_path = "C:/Users/HP/OneDrive/Documents/test dataset"

# Get a list of all CSV files in the directory
file_paths = glob.glob(directory_path + "/*.csv")

# Now file_paths contains the paths of all CSV files in the specified directory

# Read each CSV file and concatenate them into a single DataFrame
dfs = [pd.read_csv(file) for file in file_paths]
df = pd.concat(dfs)
#print(sisfall_data)
#print(sisfall_data.info())

# Remove leading whitespace characters from column names
df.columns = df.columns.str.strip()

# Calculate threshold
magnitude_data = df[['X-Axis', 'Y-Axis', 'Z-Axis']].apply(np.linalg.norm, axis=1)  # Calculate magnitude
mean = np.mean(magnitude_data)
std_dev = np.std(magnitude_data)
threshold = mean + 3 * std_dev

print("Threshold:", threshold)

# Step 2: Prepare features and labels
X = df[['X-Axis', 'Y-Axis', 'Z-Axis']]  # Features
y = df['activity_type']  # Labels

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest model with a specified number of trees
import matplotlib.pyplot as plt

# Step 2: Prepare features and labels
X = df[['X-Axis', 'Y-Axis', 'Z-Axis']]  # Features
y = df['activity_type']  # Labels

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists to store number of trees and corresponding accuracies
n_estimators_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
accuracies = []

# Train the Random Forest model with different number of trees and record accuracies
for n_estimators in n_estimators_list:
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plot the accuracy vs. number of trees
plt.plot(n_estimators_list, accuracies, marker='o')
plt.title('Accuracy vs. Number of Trees in Random Forest')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC

# Load a sample dataset (you can replace this with your own dataset)
digits = load_digits()
X, y = digits.data, digits.target

# Create a support vector classifier
estimator = SVC(gamma=0.001)

# Create learning curve
train_sizes, train_scores, test_scores, fit_times, _ = \
    learning_curve(estimator, X, y, cv=5, n_jobs=-1,
                   train_sizes=np.linspace(.1, 1.0, 5),
                   return_times=True)

# Calculate mean and standard deviation of test scores
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve for cross-validation score only
plt.figure()
plt.title("Learning Curve (Cross-validation Score)")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.grid()

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color="g")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()


# Step 5: Test the Random Forest model
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


