
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="ReFill Hub Dashboard", layout="wide")

section = st.sidebar.selectbox(
    "Navigate",
    ["Logo & Tagline", "About Company", "Analysis", "Team Members"]
)

if section == "Logo & Tagline":
    st.image("refillhub_logo.png", width=250)
    st.markdown("<h2 style='text-align:center;'>Refill Hub – Smart Sustainability, Smart Living</h2>", unsafe_allow_html=True)

elif section == "About Company":
    st.header("About ReFill Hub")
    st.write("""
    ReFill Hub is a smart sustainability initiative that enables users to refill daily essentials 
    using automated kiosks. Our mission is to reduce plastic waste and promote an eco-friendly 
    circular economy through smart technology.
    """)

elif section == "Analysis":
    st.header("Data Analysis Dashboard")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Dataset Overview")
        st.dataframe(df.head())

        st.write("This table shows the first few rows of your dataset, giving a quick snapshot of column names and data structure. It helps you verify that your uploaded file is correctly formatted and ready for analysis.")

        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=['object']).columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

        y = df_enc.iloc[:, -1]
        X = df_enc.iloc[:, :-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.subheader("Model Performance")

        # Two graphs side by side
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Feature Importance**: Shows which variables impact predictions the most.")
            fig, ax = plt.subplots(figsize=(4,3))
            importances = model.feature_importances_
            ax.barh(X.columns, importances)
            st.pyplot(fig)

        with col2:
            st.write("**Prediction Distribution**: Shows how many predictions fall into each category.")
            fig2, ax2 = plt.subplots(figsize=(4,3))
            pd.Series(preds).value_counts().plot(kind='bar', ax=ax2)
            st.pyplot(fig2)

        st.write("### Classification Report")
        st.text(classification_report(y_test, preds))

elif section == "Team Members":
    st.header("Our Team")
    st.write("👑 **Nishtha – Insights Lead**")
    st.write("✨ **Anjali – Data Analyst**")
    st.write("🌱 **Amatulla – Sustainability Research**")
    st.write("📊 **Amulya – Analytics Engineer**")
    st.write("🧠 **Anjan – Strategy & AI**")
