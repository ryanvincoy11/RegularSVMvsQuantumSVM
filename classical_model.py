import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="rbf")

start = time.perf_counter()

model.fit(X_train, y_train)

end = time.perf_counter()
classic_training_time = end - start

predictions = model.predict(X_test)

classic_accuracy = accuracy_score(y_test, predictions)

#print(f"Classical Accuracy: {classic_accuracy:.4f}")

#print("\nConfusion Matrix:")
classic_conf_matrix = confusion_matrix(y_test, predictions)
#print(classic_conf_matrix)

#print("\nClassification Report:")
classic_report = classification_report(
y_test,
predictions,
target_names=iris.target_names
)
#print(classic_report)

#print(f"Training Time: {classic_training_time:.4f} seconds")
