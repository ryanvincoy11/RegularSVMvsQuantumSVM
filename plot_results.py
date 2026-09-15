import matplotlib.pyplot as plt
from classical_model import accuracy as classical_acc
from quantum_model import accuracy as quantum_acc

models = ["Classical", "Quantum"]
scores = [classical_acc, quantum_acc]

plt.bar(models, scores)

plt.title("Iris Classification Accuracy")

plt.ylabel("Accuracy")

plt.ylim(0.5, 1.0)

plt.show()