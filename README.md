# 🍷 Wine Quality Optimization Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red)
![Optuna](https://img.shields.io/badge/Optuna-Success-green)

A professional Machine Learning dashboard demonstrating real-time hyperparameter optimization using **Streamlit** and **Optuna**. This project visualizes the tuning process of a Random Forest Classifier on the classic Wine dataset.

## ✨ Key Features

- **Curated Datasets**: One-click access to popular datasets (Titanic, Heart Disease) - no manual download required.
- **Brain (Optuna)**: Uses Tree-structured Parzen Estimator (TPE) algorithm for intelligent search.
- **Interactive UI**: Real-time updates of Accuracy, Precision, Recall, and F1 Score.
- **Modular Architecture**: Clean code structure separation Logic, Data, and Presentation.
- **Generic CSV Support**: Upload any tabular classification dataset.

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "Classification Models"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure

```text
.
├── app.py                  # Main Application (UI Layer)
├── requirements.txt        # Dependencies
└── src/
    ├── __init__.py
    ├── data.py             # Data Processing & Noise Injection
    ├── optimization.py     # Optuna Wrapper & Objective Function
    └── visualization.py    # Plotly Chart Generation
```

## 🛠️ Technologies
- **Frontend**: Streamlit
- **ML Engine**: Scikit-Learn
- **Optimization**: Optuna
- **Visualization**: Plotly

---
*Created by [Your Name] - 2024*
