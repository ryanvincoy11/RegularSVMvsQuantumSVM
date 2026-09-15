import time

from classical_model import accuracy as classical_acc
from quantum_model import accuracy as quantum_acc

print()
print("RESULTS")
print("------------------------")
print(f"Classical: {classical_acc:.4f}")
print(f"Quantum:   {quantum_acc:.4f}")