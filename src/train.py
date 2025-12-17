import mlflow
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

X, y = [], []

for file in os.listdir("data/processed"):
    X.append(np.load(os.path.join("data/processed", file)).flatten())
    y.append(1 if "correct" in file else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.start_run()

model = SVC(kernel="rbf")
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

mlflow.log_param("model", "SVM")
mlflow.log_param("kernel", "rbf")
mlflow.log_metric("accuracy", acc)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
mlflow.log_artifact("models/model.pkl")

mlflow.end_run()

print("Training selesai | Accuracy:", acc)
