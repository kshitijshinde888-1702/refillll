
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlxtend.frequent_patterns import apriori, association_rules

# Theme colors
GREEN="#0f5132"
DARK="#0a3622"
LIGHT="#d8f3dc"

st.set_page_config(page_title="ReFill Hub – Eco Analytics", layout="wide")

# Load logo
if os.path.exists("refillhub_logo.png"):
    st.image("refillhub_logo.png", width=160)

# Load dataset
df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")

# Sidebar
st.sidebar.title("🌱 What do you want to see?")
page = st.sidebar.radio("", ["🏠 Dashboard Overview", "🧩 About ReFill Hub", "📊 Analysis"])

# Dashboard Overview
if page == "🏠 Dashboard Overview":
    st.markdown(f"<h1 style='color:{GREEN}; font-weight:800;'>ReFill Hub – Eco Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.write("Sustainable analytics for refill innovation in the UAE.")

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Responses", len(df))
    c2.metric("Features", df.shape[1])
    c3.metric("Eco Personas", "3+")

# About Page
elif page == "🧩 About ReFill Hub":
    st.markdown(f"<h1 style='color:{GREEN}; font-weight:800;'>🌿 About ReFill Hub</h1>", unsafe_allow_html=True)
    st.write("""ReFill Hub promotes refill-based consumption to reduce plastic waste. 
    Our stations allow customers to refill essential household products at lower cost 
    and with lower environmental impact.""")

    st.markdown(f"<h2 style='color:{GREEN};'>👥 Team Members</h2>", unsafe_allow_html=True)
    st.write("""
    👑 Nishtha – Insights Lead  
    ✨ Anjali – Data Analyst  
    🌱 Amatulla – Sustainability Research  
    📊 Amulya – Analytics Engineer  
    🧠 Anjan – Strategy & AI  
    """)

# Analysis Page
elif page == "📊 Analysis":
    module = st.selectbox("Choose Analysis Module", [
        "Dataset Overview", "EDA", "Clustering", "Classification", "Regression", "Association Rules", "Insights"
    ])

    # Dataset Overview
    if module == "Dataset Overview":
        st.subheader("📁 Dataset Overview")
        st.dataframe(df.head())
        st.write(df.describe(include='all'))

    # EDA
    elif module == "EDA":
        st.subheader("📈 Exploratory Data Analysis")
        num_cols = df.select_dtypes(include=['int64','float64'])
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(5,3))
            ax.hist(df[col], bins=20, color=GREEN)
            ax.set_title(col, color=GREEN)
            st.pyplot(fig)
            st.write(f"Distribution of {col}.")

    # Clustering
    elif module == "Clustering":
        st.subheader("🧩 Customer Clustering")
        k = st.slider("Choose number of clusters", 2, 10, 3)
        run = st.button("Run Clustering")
        if run:
            numeric = df.select_dtypes(include=['float64','int64'])
            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric)
            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled)
            st.dataframe(df[['Cluster'] + list(numeric.columns)].head())

            fig, ax = plt.subplots(figsize=(5,3))
            ax.scatter(scaled[:,0], scaled[:,1], c=df['Cluster'], cmap="Greens", s=50)
            ax.set_title("Cluster Map", color=GREEN)
            st.pyplot(fig)
            st.write("Eco-consumer persona clustering.")

    # Classification
    elif module == "Classification":
        st.subheader("🤖 Classification Models")

        df_c = df.copy()
        le = LabelEncoder()
        df_c['Likely_to_Use_ReFillHub'] = le.fit_transform(df_c['Likely_to_Use_ReFillHub'])
        X = df_c.select_dtypes(include=['float64','int64']).drop(columns=['Likely_to_Use_ReFillHub'])
        y = df_c['Likely_to_Use_ReFillHub']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        for name, model in models.items():
            st.markdown(f"### 🌿 {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
            st.pyplot(fig)

            report = classification_report(y_test, y_pred, output_dict=True)
            weighted = report['weighted avg']
            df_report = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1-Score', 'Support', 'Accuracy'],
                'Value': [
                    weighted['precision'],
                    weighted['recall'],
                    weighted['f1-score'],
                    weighted['support'],
                    report['accuracy']
                ]
            })
            st.write("Weighted Avg Performance")
            st.dataframe(df_report)

    # Regression
    elif module == "Regression":
        st.subheader("💵 Regression – Willingness To Pay (AED)")
        target="Willingness_to_Pay_AED"
        df_r=df.dropna(subset=[target])
        X=df_r.select_dtypes(include=['float64','int64']).drop(columns=[target])
        y=df_r[target]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        model=LinearRegression()
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.write("MAE:", mean_absolute_error(y_test,y_pred))
        st.write("RMSE:", np.sqrt(mean_squared_error(y_test,y_pred)))
        st.write("Eco consumer payment prediction.")

    # Association Rules
    elif module == "Association Rules":
        st.subheader("🔗 Association Rule Mining")
        cat=df.select_dtypes(include=['object']).fillna("Missing")
        onehot=pd.get_dummies(cat)
        freq=apriori(onehot,min_support=0.1,use_colnames=True)
        rules=association_rules(freq,metric="lift",min_threshold=1)
        rules_clean=rules[['antecedents','consequents','support','confidence','lift']]
        rules_clean['antecedents']=rules_clean['antecedents'].apply(lambda x:', '.join(list(x)))
        rules_clean['consequents']=rules_clean['consequents'].apply(lambda x:', '.join(list(x)))
        st.dataframe(rules_clean.sort_values('lift',ascending=False).head(10))

    # Insights
    elif module == "Insights":
        st.subheader("💡 Insights")
        st.write("Eco-aware consumers show highest refill likelihood.")
        st.write("Sustainability motivation drives price sensitivity.")
        st.write("Cluster personas help target eco segments better.")
