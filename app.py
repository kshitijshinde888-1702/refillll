
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlxtend.frequent_patterns import apriori, association_rules

# Page settings
st.set_page_config(page_title="ReFill Hub – Black Gold Dashboard", layout="wide")

# Load data
df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")

# Sidebar Navigation
st.sidebar.markdown("## 🟨 ReFill Hub Navigation")
page = st.sidebar.radio("Choose Page:", [
    "🏠 Home",
    "📁 Dataset Overview",
    "📈 EDA",
    "🧩 Clustering",
    "🤖 Classification",
    "📉 Regression (WTP)",
    "🔗 Association Rules",
    "💡 Insights"
])

# Logo
if os.path.exists("refillhub_logo_gold.png"):
    st.sidebar.image("refillhub_logo_gold.png", width=180)

# GOLD STYLE CSS
st.markdown("""
<style>
body {
    background-color: black;
}
h1, h2, h3, h4, h5 {
    color: #D4AF37 !important;
}
</style>
""", unsafe_allow_html=True)

# HOME PAGE
if page == "🏠 Home":
    st.image("refillhub_logo_gold.png", width=260)
    st.markdown("<h1 style='color:#D4AF37;'>ReFill Hub Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:white; font-size:18px;'>
    Welcome to the premium ReFill Hub Black–Gold Intelligence Dashboard.<br>
    Explore powerful analytics, clustering, prediction, and rule mining insights.
    </p>
    """, unsafe_allow_html=True)

# DATASET OVERVIEW
elif page == "📁 Dataset Overview":
    st.markdown("## 📁 Dataset Overview")
    st.dataframe(df.head())
    st.write(df.describe(include='all'))

# EDA
elif page == "📈 EDA":
    st.markdown("## 📈 Exploratory Data Analysis")
    num_cols = df.select_dtypes(include=['int64','float64'])
    for col in num_cols:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20, color="#D4AF37")
        ax.set_title(col, color="#D4AF37")
        st.pyplot(fig)

# CLUSTERING
elif page == "🧩 Clustering":
    st.markdown("## 🧩 Advanced Clustering – Black Gold Edition")
    k = st.slider("Select number of clusters", 2, 10, 3)
    run = st.button("Run Clustering")
    if run:
        numeric = df.select_dtypes(include=['float64','int64'])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled)
        st.success(f"Clustering completed with {k} clusters!")
        st.dataframe(df[['Cluster'] + list(numeric.columns)].head())
        fig, ax = plt.subplots()
        ax.scatter(scaled[:,0], scaled[:,1], c=df['Cluster'], cmap="tab10")
        st.pyplot(fig)

# CLASSIFICATION
elif page == "🤖 Classification":
    st.markdown("## 🤖 Classification – Likely to Use ReFill Hub")
    df_class = df.copy()
    le = LabelEncoder()
    df_class['Likely_to_Use_ReFillHub'] = le.fit_transform(df_class['Likely_to_Use_ReFillHub'])
    X = df_class.select_dtypes(include=['float64','int64']).drop(columns=['Likely_to_Use_ReFillHub'])
    y = df_class['Likely_to_Use_ReFillHub']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=RandomForestClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    fig,ax=plt.subplots()
    sns.heatmap(cm,annot=True,fmt="d",cmap="YlOrBr")
    st.pyplot(fig)
    st.text(classification_report(y_test,y_pred))

# REGRESSION
elif page == "📉 Regression (WTP)":
    st.markdown("## 📉 Regression – Willingness To Pay (AED)")
    target="Willingness_to_Pay_AED"
    df_reg=df.copy().dropna(subset=[target])
    X=df_reg.select_dtypes(include=['float64','int64']).drop(columns=[target])
    y=df_reg[target]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    st.write("MAE:", mean_absolute_error(y_test,y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test,y_pred)))

# ASSOCIATION RULES
elif page == "🔗 Association Rules":
    st.markdown("## 🔗 Association Rule Mining")
    cat=df.select_dtypes(include=['object']).fillna("Missing")
    onehot=pd.get_dummies(cat)
    freq=apriori(onehot,min_support=0.1,use_colnames=True)
    rules=association_rules(freq,metric="lift",min_threshold=1)
    rules_clean=rules[['antecedents','consequents','support','confidence','lift']]
    rules_clean['antecedents']=rules_clean['antecedents'].apply(lambda x:', '.join(list(x)))
    rules_clean['consequents']=rules_clean['consequents'].apply(lambda x:', '.join(list(x)))
    st.dataframe(rules_clean.sort_values('lift',ascending=False).head(10))

# INSIGHTS
elif page == "💡 Insights":
    st.markdown("## 💡 Strategic Insights")
    st.markdown("<p style='color:#D4AF37;'>Premium AI insights for ReFill Hub decision-makers.</p>", unsafe_allow_html=True)
