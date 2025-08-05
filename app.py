import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Load dataset
df = pd.read_csv('data/titanic.csv')

# Page configuration
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

# Sidebar navigation
section = st.sidebar.radio("Navigation", ["Overview", "Data Exploration", "Visualizations", "Make Prediction", "Model Performance"])

# Overview
if section == "Overview":
    st.title("🚢 Titanic Survival Prediction App")
    st.markdown("""
    This app predicts passenger survival on the Titanic using a machine learning model trained on the Titanic dataset.
    
    **Features:**
    - Explore the dataset
    - View interactive visualizations
    - Make predictions
    - See model performance
    """)

# Data Exploration
elif section == "Data Exploration":
    st.header("🔍 Data Exploration")
    st.write("Shape of the dataset:", df.shape)
    st.write("Data Types:")
    st.write(df.dtypes)
    
    st.subheader("Sample Data")
    st.write(df.head())

    st.subheader("Filter by Gender")
    gender_filter = st.selectbox("Select Gender", df["Sex"].unique())
    st.write(df[df["Sex"] == gender_filter])

# Visualizations
elif section == "Visualizations":
    st.header("📊 Visualizations")

    st.subheader("Survival Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Survived', ax=ax1)
    st.pyplot(fig1)

    st.subheader("Survival by Gender")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='Sex', hue='Survived', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Age Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(df['Age'].dropna(), kde=True, bins=30, ax=ax3)
    st.pyplot(fig3)

# Prediction
elif section == "Make Prediction":
    st.header("🎯 Predict Survival")
    st.write("Enter the details below to get a prediction.")

    Pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
    Sex_male = st.radio("Gender", ['Male', 'Female']) == 'Male'
    Age = st.slider("Age", 1, 80, 30)
    SibSp = st.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
    Parch = st.slider("Number of Parents/Children Aboard", 0, 6, 0)
    Fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2)
    Embarked_Q = st.checkbox("Embarked at Q?")
    Embarked_S = st.checkbox("Embarked at S?")

    input_data = np.array([[Pclass, Age, SibSp, Parch, Fare, int(Sex_male), int(Embarked_Q), int(Embarked_S)]])

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        st.success(f"Predicted: {'Survived' if prediction == 1 else 'Did Not Survive'}")
        st.info(f"Prediction Confidence: {round(proba[prediction] * 100, 2)}%")

# Model Performance
elif section == "Model Performance":
    st.header("📈 Model Evaluation")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Preprocess for evaluation
    df_copy = df.copy()
    df_copy.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)
    df_copy['Age'].fillna(df_copy['Age'].median(), inplace=True)
    df_copy['Embarked'].fillna(df_copy['Embarked'].mode()[0], inplace=True)
    df_copy = pd.get_dummies(df_copy, columns=['Sex', 'Embarked'], drop_first=True)

    X = df_copy.drop('Survived', axis=1)
    y = df_copy['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    st.subheader("Accuracy")
    st.write(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
