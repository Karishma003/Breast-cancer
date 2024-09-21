# streamlit_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Set up the app
st.title("Breast Cancer Classification App")
st.sidebar.title("Settings")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("breast-cancer[1].csv")
    return data

data = load_data()

# Display data preview
st.write("### Data Preview")
st.dataframe(data.head())

# Check for missing values
st.write("### Check for Missing Values")
st.write(data.isnull().sum())

# Encode diagnosis column
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Visualize correlation heatmap
st.write("### Correlation Heatmap")
if st.sidebar.checkbox("Show Correlation Heatmap", False):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    correlation = numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", linewidths=0.5)
    st.pyplot(plt)

# Train-Test Split
X = data.drop(columns="diagnosis")
Y = data["diagnosis"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Hyperparameter Tuning using RandomizedSearchCV
def train_random_forest(X_train, Y_train):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_dist = {
        'n_estimators': [100, 200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                       n_iter=100, scoring='roc_auc', cv=3, 
                                       verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_train, Y_train)
    return random_search.best_estimator_

if st.sidebar.button("Train Model"):
    with st.spinner("Training the Random Forest model..."):
        best_rf = train_random_forest(X_train, Y_train)
        Y_pred = best_rf.predict(X_test)
        Y_prob = best_rf.predict_proba(X_test)[:, 1]

        # Display classification report and confusion matrix
        st.write("### Classification Report")
        st.text(classification_report(Y_test, Y_pred))

        st.write("### Confusion Matrix")
        st.write(confusion_matrix(Y_test, Y_pred))

        # Display AUC-ROC Score
        roc_auc = roc_auc_score(Y_test, Y_prob)
        st.write(f"AUC-ROC Score: {roc_auc}")

        # Display best hyperparameters
        st.write("### Best Hyperparameters")
        st.write(best_rf.get_params())

# Footer
st.write("Developed using Streamlit")
