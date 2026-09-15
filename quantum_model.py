import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

iris = load_iris()

X = iris.data
y = iris.target

# Use only first two features initially
X = X[:, :4]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Scale between 0 and pi
scaler = MinMaxScaler(feature_range=(0, 3.14159))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

feature_map = ZZFeatureMap(
    feature_dimension=4,
    reps=2
)

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

model = SVC(kernel=quantum_kernel.evaluate)

start = time.perf_counter()

model.fit(X_train, y_train)

end = time.perf_counter()
training_time = end - start

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Quantum Accuracy: {accuracy:.4f}")

print("/nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
y_test,
predictions,
target_names=iris.target_names
))

print(f"Training Time: {training_time:.4f} seconds")