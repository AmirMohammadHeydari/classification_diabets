# 🩺 Diabetes Classification using Support Vector Machine

A professional Machine Learning pipeline for diabetes prediction using Support Vector Machine (SVM), featuring automated preprocessing, hyperparameter optimization, model evaluation, and exploratory data analysis.

---

## 📌 Project Overview

This project demonstrates a complete end-to-end Machine Learning workflow using the Pima Indians Diabetes Dataset.

Instead of focusing only on model training, the project emphasizes clean software architecture and reproducible machine learning practices.

The pipeline includes:

- Exploratory Data Analysis (EDA)
- Feature Transformation
- Multiple preprocessing strategies
- Hyperparameter Optimization
- Model Comparison
- Performance Evaluation
- Visualization
- Model Persistence

---

## 📂 Project Structure

```
diabetes-classification/
│
├── README.md
├── requirements.txt
├── config.py
├── train.py
├── evaluate.py
├── predict.py
│
├── data/
│   └── diabetes.csv
│
├── models/
│
├── results/
│   ├── figures/
│   ├── reports/
│   └── metrics/
│
└── src/
    ├── data_loader.py
    ├── eda.py
    ├── preprocessing.py
    ├── visualization.py
    ├── trainer.py
    ├── evaluator.py
    └── utils.py
```

---

# ⚙ Features

✔ Modular project architecture

✔ Object-Oriented Design (OOP)

✔ Exploratory Data Analysis

✔ Automatic Outlier Detection (IQR)

✔ Log Transformation

✔ StandardScaler

✔ MinMaxScaler

✔ Scikit-Learn Pipeline

✔ GridSearchCV

✔ Cross Validation

✔ Model Comparison

✔ Confusion Matrix

✔ Classification Report

✔ ROC-AUC

✔ Automatic Figure Generation

✔ Automatic Report Generation

✔ Model Serialization

---

# 📊 Dataset

Dataset:

**Pima Indians Diabetes Database**

Features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target:

- Outcome

---

# 🔬 Machine Learning Pipeline

```
Load Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Outlier Detection
      │
      ▼
Log Transformation
      │
      ▼
Train/Test Split
      │
      ▼
Pipeline
(StandardScaler / MinMaxScaler)
      │
      ▼
Support Vector Machine
      │
      ▼
GridSearchCV
      │
      ▼
Best Model
      │
      ▼
Evaluation
```

---

# 🧠 Models

Four different experiments are performed.

| Experiment | Data | Scaler |
|------------|------|---------|
| Model 1 | Original | StandardScaler |
| Model 2 | Original | MinMaxScaler |
| Model 3 | Log Transformed | StandardScaler |
| Model 4 | Log Transformed | MinMaxScaler |

---

# 📈 Evaluation Metrics

The project reports

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

---

# 📷 Generated Outputs

The project automatically generates

```
results/

├── figures/

│   ├── histograms.png

│   ├── boxplot.png

│   ├── correlation_heatmap.png

│   ├── scatter_matrix.png

│   └── confusion_matrix.png

├── reports/

│   ├── eda_report.txt

│   └── classification_report.txt

└── metrics/
```

---

# 🚀 Installation

```bash
git clone https://github.com/your_username/diabetes-classification.git

cd diabetes-classification

pip install -r requirements.txt
```

---

# ▶ Train

```bash
python train.py
```

---

# 📊 Evaluate

```bash
python evaluate.py
```

---

# 🔮 Predict

```bash
python predict.py --input sample.csv
```

---

# 🛠 Technologies

- Python

- Pandas

- NumPy

- Scikit-Learn

- Matplotlib

- Joblib

---

# 📖 Software Design

The project follows modern software engineering practices:

- Modular Architecture

- Single Responsibility Principle

- Separation of Concerns

- Object-Oriented Programming

- Reproducible Machine Learning

- Config-driven Design

---

# 📌 Future Improvements

- XGBoost

- LightGBM

- CatBoost

- SHAP Explainability

- FastAPI Deployment

- Docker Support

- MLflow Experiment Tracking

- Unit Testing

---

# 👨‍💻 Author

Your Name

GitHub:
https://github.com/AmirMohammadHeydari

LinkedIn:
www.linkedin.com/in/heidary-amirmohammad-955942418
