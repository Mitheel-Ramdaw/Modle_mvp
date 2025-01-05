import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from math import pi

# Define data for grades
grading_data = {
    "Grade 2": {
        "sections": {
            "Number Recognition and Counting": 5,
            "Basic Operations": 5,
            "Shapes and Space": 5,
            "Measurement and Data Handling": 5,
        },
        "total_marks": 20,
        "performance_bands": {
            "Strong": (16, 20, "75–100%"),
            "Good": (12, 15, "50–74%"),
            "Support Needed": (8, 11, "30–49%"),
            "Intervention Needed": (0, 7, "<30%"),
        },
    },
    "Grade 3": {
        "sections": {
            "Numbers and Counting": 10,
            "Basic Operations": 10,
            "Shapes and Space": 10,
        },
        "total_marks": 30,
        "performance_bands": {
            "Strong": (25, 30, "75–100%"),
            "Good": (15, 24, "50–74%"),
            "Support Needed": (10, 14, "30–49%"),
            "Intervention Needed": (0, 9, "<30%"),
        },
    },
    "Grade 4": {
        "sections": {
            "Numbers and Counting": 10,
            "Basic Operations": 15,
            "Shapes and Space": 10,
            "Problem Solving": 5,
        },
        "total_marks": 40,
        "performance_bands": {
            "Strong": (35, 40, "75–100%"),
            "Good": (25, 34, "50–74%"),
            "Support Needed": (15, 24, "30–49%"),
            "Intervention Needed": (0, 14, "<30%"),
        },
    },
    "Grade 5": {
        "sections": {
            "Numbers and Counting": 20,
            "Basic Operations": 20,
            "Shapes and Space": 10,
            "Problem Solving": 10,
        },
        "total_marks": 50,
        "performance_bands": {
            "Strong": (45, 50, "75–100%"),
            "Good": (35, 44, "50–74%"),
            "Support Needed": (20, 34, "30–49%"),
            "Intervention Needed": (0, 19, "<30%"),
        },
    },
    "Grade 6": {
        "sections": {
            "Numbers and Counting": 15,
            "Basic Operations": 15,
            "Fractions and Decimals": 10,
            "Shapes and Space": 10,
            "Data Handling": 5,
            "Problem Solving": 10,
        },
        "total_marks": 60,
        "performance_bands": {
            "Strong": (50, 60, "75–100%"),
            "Good": (40, 49, "50–74%"),
            "Support Needed": (30, 39, "30–49%"),
            "Intervention Needed": (0, 29, "<30%"),
        },
    },
}

# Function to create a Ridge Regression model
def create_sample_model():
    np.random.seed(42)
    X = np.random.randint(50, 101, size=(100, 6))  # Simulated section scores
    y = np.random.randint(50, 101, size=100)  # Year-end performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    return model, scaler

ridge_model, scaler = create_sample_model()

# Streamlit Sidebar
st.sidebar.title("Diagnostic Test Predictor")
grade = st.sidebar.selectbox("Select Grade", list(grading_data.keys()))
st.sidebar.write(f"You selected: {grade}")

# Fetch grade-specific data
grade_data = grading_data[grade]
sections = grade_data["sections"]
total_marks = grade_data["total_marks"]
performance_bands = grade_data["performance_bands"]

# Input Scores
st.title(f"{grade} Diagnostic Test")
st.header("Enter Diagnostic Scores")
scores = {}
for section, max_marks in sections.items():
    scores[section] = st.slider(f"{section} (Max {max_marks} Marks)", 0, max_marks, max_marks // 2)

# Calculate Total Score
total_score = sum(scores.values())
st.subheader("Results")
st.write(f"Total Diagnostic Score: {total_score} / {total_marks}")

# Determine Performance Band
predicted_band = None
for band, (min_score, max_score, range_desc) in performance_bands.items():
    if min_score <= total_score <= max_score:
        predicted_band = f"{band} ({range_desc})"
        break

st.write(f"Predicted Performance Band: {predicted_band}")

# Predict Year-End Performance
input_data = np.array([list(scores.values()) + [0] * (6 - len(scores))])
input_scaled = scaler.transform(input_data)
predicted_year_end = ridge_model.predict(input_scaled)[0]
st.write(f"Predicted Year-End Performance: {predicted_year_end:.2f}%")

# Radar Chart
st.subheader("Section Scores Visualization")

categories = list(sections.keys())
values = list(scores.values())
values += values[:1]  # Close the radar chart loop

angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]  # Add the first angle to close the loop

fig, ax = plt.subplots(subplot_kw={"polar": True})
ax.fill(angles, values, color="blue", alpha=0.25)
ax.plot(angles, values, color="blue", linewidth=2)
ax.set_yticks([max(sections.values()) / 4, max(sections.values()) / 2, max(sections.values()) * 3 / 4])
ax.set_yticklabels(["Low", "Medium", "High"], fontsize=8)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title("Diagnostic Section Scores")
st.pyplot(fig)
