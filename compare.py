from classical_model import (
    classic_accuracy,
    classic_conf_matrix,
    classic_report,
    classic_training_time,
)

from quantum_model import (
    quantum_accuracy,
    quantum_conf_matrix,
    quantum_report,
    quantum_training_time,
)

print()
print("RESULTS")
print("------------------------")
print(f"Classical Accuracy: {classic_accuracy:.4f}")
print(f"Quantum Accuracy:   {quantum_accuracy:.4f}")

print("------------------------")
print("Classical Confusion Matrix:")
print(classic_conf_matrix)

print("\nQuantum Confusion Matrix:")
print(quantum_conf_matrix)

print("------------------------")
print("Classical Classification Report:")
print(classic_report)

print("\nQuantum Classification Report:")
print(quantum_report)

print("------------------------")
print(f"Classical Training Time: {classic_training_time:.4f} seconds")
print(f"Quantum Training Time:   {quantum_training_time:.4f} seconds")

speed_ratio = quantum_training_time / classic_training_time

print("------------------------")
print(f"Quantum model was approximately {speed_ratio:.0f}x slower.")