import csv
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
data = []
labels = []

with open("asl_dataset.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row

    for row in reader:
        labels.append(row[0])
        data.append([float(value) for value in row[1:]])

X = np.array(data)
y = np.array(labels)

print("Total samples:", len(X))
print("Number of features:", X.shape[1])

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create and train the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions))

# 5. Save the trained model
joblib.dump(model, "asl_model.pkl")

print("\nModel saved as asl_model.pkl")