# Regular SVM vs Quantum SVM on the Iris Dataset

## Overview

This project compares a classical Support Vector Machine (SVM) with a Quantum Kernel Support Vector Machine (QSVM) using the Iris flower dataset.

The goal is to investigate whether a quantum feature representation can improve classification performance compared to a traditional machine learning approach.

This project evaluates both approaches using common machine learning metrics and compares their computational efficiency.

---

## Dataset

The Iris dataset contains 150 flower samples from three species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

Each flower contains four measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The objective is to predict the species of a flower using these measurements.

---

## Models

### Classical Model

The classical model uses:

- StandardScaler
- Support Vector Machine (SVM)
- Radial Basis Function (RBF) Kernel

Pipeline:

```text
Iris Data
    ↓
StandardScaler
    ↓
SVM (RBF Kernel)
    ↓
Prediction
```

### Quantum Model

The quantum model uses:

- ZZFeatureMap
- FidelityQuantumKernel
- Support Vector Machine (SVM)

Pipeline:

```text
Iris Data
    ↓
Quantum Feature Map
    ↓
Quantum Kernel
    ↓
   SVM
    ↓
Prediction
```

---

## What is a Quantum Kernel?

A kernel measures the similarity between two data points.

In a classical SVM, similarity is calculated directly from the feature vectors.

In the quantum model, data is first encoded into quantum states using a quantum circuit. The similarity between these quantum states is then calculated and used by the SVM.

Conceptually:

```text
Flower Features
      ↓
Quantum Circuit
      ↓
Quantum State
      ↓
Similarity Calculation
      ↓
SVM
      ↓
Prediction
```

The purpose of the quantum kernel is to transform the data into a higher-dimensional feature space where classes may become easier to separate.

---

## Evaluation Metrics

The following metrics are used to evaluate both models:

### Accuracy

Measures the percentage of correct predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how often a predicted class is correct.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual examples of a class were successfully identified.

```text
Recall = TP / (TP + FN)
```

### F1 Score

Balances precision and recall.

```text
F1 = 2 × (Precision × Recall)
     ------------------------
      Precision + Recall
```

### Confusion Matrix

Displays actual classifications versus predicted classifications and helps identify where mistakes occur.

### Training Time

Measures how long each model takes to train.

---

## Technologies Used

- Python
- Scikit-Learn
- Qiskit
- Qiskit Machine Learning
- Matplotlib

---

## Project Structure

```text
iris_qml/
│
├── classical_model.py
├── quantum_model.py
├── compare.py
├── plot_results.py
├── README.md
└── requirements.txt
```

---

## Running the Project

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Classical Model

```bash
python classical_model.py
```

### Run the Quantum Model

```bash
python quantum_model.py
```

### Compare Results

```bash
python compare.py
```

---

## Research Question

Can a quantum feature representation improve classification performance compared to a traditional Support Vector Machine on the Iris dataset?

---

## Future Improvements

- Compare multiple quantum feature maps
- Compare multiple SVM kernels
- Add confusion matrix visualizations
- Add performance charts
- Build a React dashboard for model comparison
- Test larger datasets
- Evaluate performance on real quantum hardware

---

## Author

Ryan Vincoy
