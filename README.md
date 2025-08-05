
# 🚢 Titanic Survival Prediction Web App

This project predicts whether a passenger survived the Titanic disaster using a machine learning model built with the Titanic dataset. The app is built with Streamlit and deployed to Streamlit Cloud.

---

## 📊 Dataset

- Source: [Kaggle - Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- Target variable: `Survived` (1 = Yes, 0 = No)

---

## ⚙️ Features

- **Data Exploration**: View dataset structure and apply filters.
- **Visualizations**: Analyze survival distribution using interactive plots.
- **Make Predictions**: Input passenger details and get a survival prediction.
- **Model Performance**: See accuracy, confusion matrix, and classification report.

---

## 🧪 Model

- Trained with `RandomForestClassifier` from scikit-learn.
- Compared with Logistic Regression.
- Saved using `pickle`.

---

## 🧾 How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/titanic-ml-app.git
   cd titanic-ml-app
