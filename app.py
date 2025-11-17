
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ReFill Hub – Dashboard", layout="wide", initial_sidebar_state="expanded")

df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Home",
        "📁 Dataset Overview",
        "📈 EDA",
        "🧩 Clustering",
        "🤖 Classification",
        "📉 Regression (WTP)",
        "🔗 Association Rules",
        "💡 Insights"
    ]
)

# ---------------- HOME PAGE ----------------
if page == "🏠 Home":
    st.title("Welcome to ReFill Hub Analytics Dashboard")

    logo_path = "refillhub_logo_intro.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=350)

    st.write("""
### 🏢 About ReFill Hub
ReFill Hub is an innovative smart–refill station concept designed to reduce plastic waste, promote sustainability, 
and provide customers with easy access to refillable eco-friendly daily essentials.

Our mission:
- ♻ Reduce single-use plastic  
- 🌍 Promote sustainable habits  
- 💡 Provide affordable refill solutions across the UAE  

Use the sidebar to explore analytics insights.
""")

# ---------------- DATASET OVERVIEW ----------------
elif page == "📁 Dataset Overview":
    st.title("📁 Dataset Overview")
    st.dataframe(df.head())
    st.write(df.describe(include='all'))

# ---------------- EDA ----------------
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")
    num_cols = df.select_dtypes(include=['int64','float64'])
    for col in num_cols:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20)
        ax.set_title(col)
        st.pyplot(fig)

# ---------------- CLUSTERING ----------------
elif page == "🧩 Clustering":
    st.title("🧩 Customer Segmentation (Clustering)")
    num = df.select_dtypes(include=['float64','int64'])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(num)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled)
    st.dataframe(df[['Cluster'] + list(num.columns)].head())
    fig, ax = plt.subplots()
    ax.scatter(scaled[:,0], scaled[:,1], c=df['Cluster'])
    st.pyplot(fig)

# ---------------- CLASSIFICATION ----------------
elif page == "🤖 Classification":
    st.title("🤖 Predict Likely to Use ReFill Hub")

    df_class = df.copy()
    le = LabelEncoder()
    df_class['Likely_to_Use_ReFillHub'] = le.fit_transform(df_class['Likely_to_Use_ReFillHub'])

    X = df_class.select_dtypes(include=['float64','int64']).drop(columns=['Likely_to_Use_ReFillHub'])
    y = df_class['Likely_to_Use_ReFillHub']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

    st.text(classification_report(y_test, y_pred))

# ---------------- REGRESSION ----------------
elif page == "📉 Regression (WTP)":
    st.title("📉 Predict Willingness to Pay (AED)")
    df_reg = df.copy()
    target = "Willingness_to_Pay_AED"
    df_reg = df_reg.dropna(subset=[target])
    X = df_reg.select_dtypes(include=['float64','int64']).drop(columns=[target])
    y = df_reg[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("MAE:", mean_absolute_error(y_test,y_pred))
    st.write("MSE:", mean_squared_error(y_test,y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test,y_pred)))

# ---------------- ASSOCIATION RULES ----------------
elif page == "🔗 Association Rules":
    st.title("🔗 Association Rule Mining")

    cat_cols = df.select_dtypes(include=['object']).fillna("Missing")
    onehot = pd.get_dummies(cat_cols)

    freq = apriori(onehot, min_support=0.1, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1)

    rules_clean = rules[["antecedents","consequents","support","confidence","lift"]].sort_values("lift", ascending=False)
    rules_clean["antecedents"] = rules_clean["antecedents"].apply(lambda x: ', '.join(list(x)))
    rules_clean["consequents"] = rules_clean["consequents"].apply(lambda x: ', '.join(list(x)))

    st.dataframe(rules_clean.head(10))

# ---------------- INSIGHTS ----------------
elif page == "💡 Insights":
    st.title("💡 Business Insights & Recommendations")
    st.write("""
### Key Insights:
- Eco-conscious customers show strongest refill interest.
- Clustering reveals three distinct customer personas.
- Social influence and sustainability awareness strongly affect adoption.
- WTP is strongly related to income, spending, and sustainability motivation.

### Recommendations:
- Position ReFill Hub in eco-aware and high-income communities.
- Introduce refill-based loyalty programs.
- Use refill bundles to increase cross-category adoption.
- Focus ads on sustainability-conscious consumers.
""")

