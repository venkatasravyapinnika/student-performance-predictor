from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "student_model.pkl"

# 2. Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# 3. Page title
st.title("🎓 Student Performance Predictor")
st.write("Enter the student's details to predict the final grade.")

# 4. Load trained model
if not MODEL_FILE.exists():
    st.error("Model not found. Please run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_FILE)

# 5. Student inputs
st.subheader("Student Details")

g1 = st.number_input(
    "First-period grade (G1) — out of 20",
    min_value=0,
    max_value=20,
    value=10,
    step=1
)

g2 = st.number_input(
    "Second-period grade (G2) — out of 20",
    min_value=0,
    max_value=20,
    value=10,
    step=1
)

studytime = st.selectbox(
    "Weekly study time",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Less than 2 hours",
        2: "2 to 5 hours",
        3: "5 to 10 hours",
        4: "More than 10 hours"
    }[x]
)

failures = st.number_input(
    "Number of previous class failures",
    min_value=0,
    max_value=4,
    value=0,
    step=1,
    help="Uses the original dataset definition, not subjects failed out of six."
)

absences = st.number_input(
    "Total school absences",
    min_value=0,
    max_value=365,
    value=0,
    step=1,
    help="Enter the total number of absences, not a percentage."
)

age = st.number_input(
    "Student age",
    min_value=15,
    max_value=22,
    value=17,
    step=1
)

# 6. Predict grade
if st.button("Predict Final Grade", type="primary"):

    # Keep feature names and order identical to train_model.py
    input_data = pd.DataFrame([{
        "G1": g1,
        "G2": g2,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "age": age
    }])

    prediction = float(model.predict(input_data)[0])

    # Keep prediction within the grade scale
    prediction = max(0.0, min(20.0, prediction))

    # 7. Display prediction
    st.success("Prediction completed!")

    st.metric(
        label="Predicted Final Grade (G3)",
        value=f"{prediction:.2f} / 20"
    )

    percentage = (prediction / 20) * 100
    st.write(f"Equivalent percentage: **{percentage:.1f}%**")

    if prediction >= 10:
        st.write("The predicted grade is at least 10 out of 20.")
    else:
        st.write("The predicted grade is below 10 out of 20.")

    # 8. Display graph
    st.subheader("📊 Student Grade Graph")

    chart_data = pd.DataFrame({
        "Grade": [
            "G1 - First Period",
            "G2 - Second Period",
            "G3 - Predicted"
        ],
        "Marks out of 20": [
            g1,
            g2,
            prediction
        ]
    })

    st.bar_chart(
        chart_data,
        x="Grade",
        y="Marks out of 20",
        y_label="Marks out of 20"
    )

    st.caption(
        "G1 and G2 are previous grades. G3 is the model's prediction."
    )

# 9. Disclaimer
st.divider()
st.caption(
    "This prediction is an estimate from a machine-learning model, "
    "not a guaranteed final grade."
)



