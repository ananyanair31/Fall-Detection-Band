import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import glob

# Step 1: Read the dataset
directory_path = "C:/Users/HP/OneDrive/Documents/test dataset"

# Get a list of all CSV files in the directory
file_paths = glob.glob(directory_path + "/*.csv")

# Read each CSV file and concatenate them into a single DataFrame
dfs = [pd.read_csv(file) for file in file_paths]
df = pd.concat(dfs)

# Remove leading whitespace characters from column names
df.columns = df.columns.str.strip()

# Step 2: Prepare features and labels
X = df[['X-Axis', 'Y-Axis', 'Z-Axis']]  # Features
y = df['activity_type']  # Labels

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest model with a specified number of trees
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

# Step 5: Create learning curve
train_sizes, train_scores, test_scores = learning_curve(rf_model, X, y, cv=5, n_jobs=-1,
                                                        train_sizes=np.linspace(.1, 1.0, 5))

# Calculate mean and standard deviation of train and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.grid()

#plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                # train_scores_mean + train_scores_std, alpha=0.1,
                 #color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color="g")
#plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         #label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()

# Step 6: Test the Random Forest model
y_pred = rf_model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
