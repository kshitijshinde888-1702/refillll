
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="ReFill Hub – Fixed Version", layout="wide")

# Load logo safely
if os.path.exists("refillhub_logo.png"):
    st.sidebar.image("refillhub_logo.png", width=150)
else:
    st.sidebar.write("Logo not found!")

st.sidebar.title("ReFill Hub – Menu")
choice = st.sidebar.radio("Navigate", ["Home", "Data Preview"])

if choice == "Home":
    st.title("ReFill Hub – Working Dashboard")
    st.write("This version fixes the FileNotFoundError by placing the logo next to app.py.")

elif choice == "Data Preview":
    df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")
    st.dataframe(df.head())
