import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
import io

# Set the page layout
st.set_page_config(page_title="Instagram Reach Analysis", layout="wide")

# Create layout with columns for centering content
left_col, center_col, right_col = st.columns([1, 2, 1])

with center_col:
    st.title("Instagram Reach Analysis")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload your Instagram data (CSV format):", type=["csv"])

    if uploaded_file:
        # Load data
        data = pd.read_csv(uploaded_file, encoding='latin1')
        st.subheader("Data Preview")
        st.dataframe(data.head())

        # Data information
        # Display dataset information in a user-friendly way
        st.subheader("Dataset Summary")
        info = {
            "Column": data.columns,
            "Non-Null Count": data.notnull().sum(),
            "Data Type": data.dtypes
        }
        summary_df = pd.DataFrame(info)
        st.table(summary_df)  # Displays dataset summary in a table format

        # For more detailed data type and memory usage info, you can display it as a markdown
        memory_usage = f"Memory Usage: {data.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB"
        st.markdown(f"**{memory_usage}**")

        # Handle missing values
        st.write("Missing Values:")
        st.write(data.isnull().sum())

        # Distribution of impressions from home
        st.subheader("Distribution of Impressions from Home")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.histplot(data['From Home'], kde=True, ax=ax, color='blue')
        ax.set_title("Distribution of Impressions from Home")
        st.pyplot(fig)

        # Distribution of impressions from hashtags
        st.subheader("Distribution of Impressions from Hashtags")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.histplot(data['From Hashtags'], kde=True, ax=ax, color='green')
        ax.set_title("Distribution of Impressions from Hashtags")
        st.pyplot(fig)

        # Distribution of impressions from explore
        st.subheader("Distribution of Impressions from Explore")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.histplot(data['From Explore'], kde=True, ax=ax, color='orange')
        ax.set_title("Distribution of Impressions from Explore")
        st.pyplot(fig)

        # Percentage of impressions from various sources
        st.subheader("Impression Sources - Percentage Distribution")
        home = data["From Home"].sum()
        hashtags = data["From Hashtags"].sum()
        explore = data["From Explore"].sum()
        other = data["From Other"].sum()

        labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
        values = [home, hashtags, explore, other]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%.0f%%', colors=sns.color_palette('pastel'))
        ax.set_title("Percentage of Impressions from Various Sources")
        st.pyplot(fig)

        # WordCloud for Captions
        st.subheader("WordCloud - Captions")
        text = " ".join(i for i in data['Caption'])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(text)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # WordCloud for Hashtags
        st.subheader("WordCloud - Hashtags")
        text = " ".join(i for i in data['Hashtags'])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(text)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # Relationships
        st.subheader("Relationships Between Variables")

        st.write("Relationship Between Likes and Impressions")
        fig = sns.lmplot(data=data, x="Impressions", y="Likes", fit_reg=True)
        st.pyplot(fig)

        st.write("Relationship Between Comments and Impressions")
        fig = sns.lmplot(data=data, x="Impressions", y="Comments", fit_reg=True)
        st.pyplot(fig)

        st.write("Relationship Between Shares and Impressions")
        fig = sns.lmplot(data=data, x="Impressions", y="Shares", fit_reg=True)
        st.pyplot(fig)

        st.write("Relationship Between Saves and Impressions")
        fig = sns.lmplot(data=data, x="Impressions", y="Saves", fit_reg=True)
        st.pyplot(fig)

        # Correlation analysis
        # Ensure only numeric columns are included for correlation
        numeric_data = data.select_dtypes(include=[np.number])

        # Calculate correlation
        correlation = numeric_data.corr()
        st.subheader("Correlation with Impressions")
        st.write(correlation["Impressions"].sort_values(ascending=False))

        # Conversion Rate
        st.subheader("Conversion Rate")
        conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
        st.write(f"Conversion Rate: {conversion_rate:.2f}%")

        # Predictive Modeling
        st.subheader("Predicting Instagram Post Reach")

        # Split data
        X = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
        y = np.array(data["Impressions"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = PassiveAggressiveRegressor()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        st.write(f"Model Accuracy: {score:.2f}")

        # Make predictions
        st.subheader("Predict Reach for a New Post")
        likes = st.number_input("Likes", min_value=0, step=1)
        saves = st.number_input("Saves", min_value=0, step=1)
        comments = st.number_input("Comments", min_value=0, step=1)
        shares = st.number_input("Shares", min_value=0, step=1)
        profile_visits = st.number_input("Profile Visits", min_value=0, step=1)
        follows = st.number_input("Follows", min_value=0, step=1)

        if st.button("Predict"):
            features = np.array([[likes, saves, comments, shares, profile_visits, follows]])
            prediction = model.predict(features)
            st.write(f"Predicted Impressions: {prediction[0]:.2f}")
