import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Personalized Learning Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# ==================================
# LOAD DATA
# ==================================

@st.cache_data
def load_data():
    return pd.read_csv("final_courses.csv")

courses = load_data()

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

tfidf_matrix = tfidf.transform(
    courses["combined_features"]
)

# ==================================
# RECOMMENDATION FUNCTION
# ==================================

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

# ==================================
# CUSTOM CSS
# ==================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.course-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f7f7f7;
    margin-bottom: 15px;
    border-left: 6px solid #4CAF50;
}

.metric-card {
    padding: 15px;
    border-radius: 10px;
    background-color: #fafafa;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ==================================
# HEADER
# ==================================

st.title("🎓 Personalized Learning Recommendation System")

st.markdown("""
Find the most relevant online courses based on your:

- Skills
- Interests
- Career Goals
- Learning Preferences

This recommendation engine uses **TF-IDF Vectorization** and **Cosine Similarity** to generate personalized learning recommendations.
""")

st.divider()

# ==================================
# SIDEBAR
# ==================================

st.sidebar.header("⚙️ Recommendation Settings")

num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

# ==================================
# PROJECT STATS
# ==================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📚 Total Courses",
        f"{len(courses):,}"
    )

with col2:
    st.metric(
        "🌐 Platforms",
        courses["platform"].nunique()
    )

with col3:
    st.metric(
        "⭐ Average Rating",
        round(
            pd.to_numeric(
                courses["rating"],
                errors="coerce"
            ).mean(),
            2
        )
    )

st.divider()

# ==================================
# USER INPUT
# ==================================

interest = st.text_area(
    "💡 Enter Your Interests",
    placeholder="""
Example:
Python
Machine Learning
Data Science
Artificial Intelligence
Cybersecurity
"""
)

# ==================================
# BUTTON
# ==================================

if st.button("🚀 Recommend Courses"):

    if interest.strip():

        recommendations = recommend_by_interest(
            interest,
            num_recommendations
        )

        st.success(
            f"Showing Top {num_recommendations} Recommendations"
        )

        for _, row in recommendations.iterrows():

            rating = row["rating"]

            try:
                rating = round(float(rating), 1)
            except:
                rating = "N/A"

            st.markdown(
                f"""
                <div class="course-card">

                <h4>📚 {row['course_title']}</h4>

                <p>
                ⭐ <b>Rating:</b> {rating}
                </p>

                <p>
                🏢 <b>Platform:</b> {row['platform']}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.warning(
            "Please enter your interests."
        )

# ==================================
# FOOTER
# ==================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    🚀 Built using Streamlit, Pandas, Scikit-Learn and Python

    <br>

    Personalized Learning Recommendation System

    </div>
    """,
    unsafe_allow_html=True
) "Please enter some interests."
        )
