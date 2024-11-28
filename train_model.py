import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Load and prepare the dataset
data = load_iris()
X, y = data.data, data.target
# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
# Save the trained model
joblib.dump(model, 'iris_model.joblib')
print("Model has been saved as 'iris_model.joblib'")