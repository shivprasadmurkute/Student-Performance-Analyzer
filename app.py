import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
log_model = joblib.load("student_pass_model.joblib")
kmeans = joblib.load("student_cluster_model.joblib")
scaler = joblib.load("student_scaler.joblib")

st.set_page_config(page_title="Student Performance Analyzer", layout="centered")
st.title("Student Performance Analyzer")

menu = st.sidebar.radio("Select Task", ["Predict Pass/Fail", "Cluster Students"])

if menu == "Predict Pass/Fail":
    st.subheader("Pass/Fail Prediction")

    uploaded_file = st.file_uploader("Upload student data (CSV)", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())

        for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
            data[col] = pd.factorize(data[col])[0]

        prediction = log_model.predict(data)
        data['Pass Prediction'] = prediction

        st.write("Prediction Results:")
        st.dataframe(data)

        st.download_button("Download Results", data.to_csv(index=False), "pass_predictions.csv")

elif menu == "Cluster Students":
    st.subheader("Student Clustering")

    math = st.slider("Math Score", 0, 100, 50)
    reading = st.slider("Reading Score", 0, 100, 50)
    writing = st.slider("Writing Score", 0, 100, 50)

    input_data = np.array([[math, reading, writing]])
    scaled = scaler.transform(input_data)
    cluster = kmeans.predict(scaled)[0]

    st.success(f"This student belongs to Cluster: {cluster}")

    # Plot cluster visualization
    df = pd.read_csv("StudentsPerformance.csv")
    scores = df[['math score', 'reading score', 'writing score']]
    scaled_scores = scaler.transform(scores)
    clusters = kmeans.predict(scaled_scores)
    df['Cluster'] = clusters

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="math score", y="writing score", hue="Cluster", palette="Set2", ax=ax)
    ax.scatter(math, writing, color="black", marker="X", s=150, label="You")
    ax.legend()
    st.pyplot(fig)
