
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ReFill Hub – Dashboard", layout="wide")

st.title("ReFill Hub – Smart Refill Stations Dashboard")

df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dataset Overview", "EDA", "Clustering",
    "Classification", "Regression", "Association Rules"
])

with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write(df.describe(include='all'))

with tab2:
    st.subheader("Exploratory Data Analysis")
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    for col in num_cols:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20)
        ax.set_title(col)
        st.pyplot(fig)

with tab3:
    st.subheader("Customer Segmentation (Clustering)")
    num = df.select_dtypes(include=['int64','float64'])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(num)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled)
    st.dataframe(df[['Cluster'] + list(num.columns)].head())
    fig, ax = plt.subplots()
    ax.scatter(scaled[:,0], scaled[:,1], c=df['Cluster'])
    st.pyplot(fig)

with tab4:
    st.subheader("Classification – Predict Likely to Use ReFillHub")

    df_class = df.copy()
    le = LabelEncoder()
    df_class['Likely_to_Use_ReFillHub'] = le.fit_transform(df_class['Likely_to_Use_ReFillHub'])

    X = df_class.select_dtypes(include=['int64','float64']).drop(columns=['Likely_to_Use_ReFillHub'])
    y = df_class['Likely_to_Use_ReFillHub']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

    st.text(classification_report(y_test, y_pred))

with tab5:
    st.subheader("Regression – Predict Willingness to Pay (AED)")
    target="Willingness_to_Pay_AED"

    df_reg = df.copy()
    df_reg = df_reg.dropna(subset=[target])

    X = df_reg.select_dtypes(include=['float64','int64']).drop(columns=[target])
    y = df_reg[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    st.write("MAE:", mean_absolute_error(y_test,y_pred))
    st.write("MSE:", mean_squared_error(y_test,y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test,y_pred)))

with tab6:
    st.subheader("Association Rule Mining")
    cat_cols = df.select_dtypes(include=['object']).fillna("Missing")
    onehot = pd.get_dummies(cat_cols)
    freq = apriori(onehot, min_support=0.1, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1)
    st.dataframe(rules.sort_values('lift',ascending=False).head(10))
