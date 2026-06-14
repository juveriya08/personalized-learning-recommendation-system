import streamlit as st
import pandas as pd
import pickle

from sklearn.metrics.pairwise import linear_kernel

# Load dataset
courses = pd.read_csv("final_courses.csv")

# Load TF-IDF model
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Create TF-IDF matrix
tfidf_matrix = tfidf.transform(
    courses["combined_features"]
)

# Recommendation function
def recommend_by_interest(interests, n=10):

    user_vector = tfidf.transform([interests])

    scores = linear_kernel(
        user_vector,
        tfidf_matrix
    ).flatten()

    top_indices = scores.argsort()[-n:][::-1]

    return courses.iloc[top_indices][
        ["course_title", "platform", "rating"]
    ]

# UI
st.title("🎓 Personalized Learning Recommendation System")

st.write(
    "Enter your interests and get personalized course recommendations."
)

interest = st.text_area(
    "Enter Skills / Interests",
    placeholder="Example: python machine learning data science"
)

if st.button("Recommend Courses"):

    if interest.strip():

        recommendations = recommend_by_interest(
            interest
        )

        st.success(
            "Top Recommended Courses"
        )

        st.dataframe(
            recommendations,
            use_container_width=True
        )

    else:
        st.warning(
            "Please enter some interests."
        )