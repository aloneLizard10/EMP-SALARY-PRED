import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import json
import csv
from datetime import datetime
import shap
import warnings
import os
warnings.filterwarnings("ignore")

# -------------------- USER LOGIN SYSTEM --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

st.title("🔐 Login to Salary Predictor")

# -------------------- ACCOUNT CREATION --------------------
def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

users = load_users()

if not st.session_state.logged_in:
    page = st.radio("Select Option", ["Login", "Create Account"], horizontal=True)

    if page == "Create Account":
        new_user = st.text_input("Choose a Username")
        new_pass = st.text_input("Choose a Password", type="password")
        if st.button("Create Account"):
            if new_user in users:
                st.warning("Username already exists.")
            elif not new_user or not new_pass:
                st.warning("Username and password cannot be empty.")
            else:
                users[new_user] = new_pass
                save_users(users)
                st.success("Account created! You can now log in.")
                st.rerun()


if not st.session_state.logged_in:
    if page == "Login":
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")

        if st.button("Login"):
            if username_input in users and users[username_input] == password_input:
                st.session_state.logged_in = True
                st.session_state.username = username_input
                st.success(f"Welcome, {username_input}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    st.stop()


# ------------------ MAIN APP AFTER LOGIN ------------------
st.title("💼 Employee Salary Prediction")
st.write(f"Hello **{st.session_state.username}**, predict your salary based on experience and certifications.")

model = joblib.load('salary_model.pkl')

# Load model metrics
try:
    with open("model_score.json") as f:
        r2 = json.load(f).get("r2_score", None)
except:
    r2 = None
    st.sidebar.warning("Model score not found.")

# Load residual std
try:
    with open("model_stats.json") as f:
        residual_std = json.load(f).get("residual_std", None)
except:
    residual_std = None

# Sidebar
st.sidebar.markdown("### 📊 Model Performance")
if r2 is not None:
    st.sidebar.write(f"**R² Score:** {r2:.2f}")

# --- Inputs ---
total_exp = st.number_input("Total Experience (years)", min_value=0.0, step=0.5)
lead_exp = st.number_input("Team Lead Experience (years)", min_value=0.0, step=0.5)
pm_exp = st.number_input("Project Manager Experience (years)", min_value=0.0, step=0.5)
certs = st.number_input("Number of Certifications", min_value=0, step=1)

# --- Predict ---
if st.button("Predict Salary"):
    features = [[total_exp, lead_exp, pm_exp, certs]]
    predicted_salary = model.predict(features)[0]
    st.success(f"💰 Estimated Salary: ₹{predicted_salary:,.2f}")

    if residual_std:
        ci = 1.96 * residual_std
        st.info(f"📉 95% Confidence Interval: ₹{predicted_salary - ci:,.2f} - ₹{predicted_salary + ci:,.2f}")

    # Save prediction to user-specific file
    filename = f"{st.session_state.username}_predictions.csv"
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Total Experience": total_exp,
        "Team Lead Experience": lead_exp,
        "Project Manager Experience": pm_exp,
        "Certifications": certs,
        "Predicted Salary": predicted_salary
    }

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=entry.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(entry)

    # Similar profiles
    st.markdown("### 🧑‍🤝‍🧑 Similar Profiles")
    try:
        df = pd.read_csv("Salary_Data.csv")
        similar = df[
            (abs(df["Total Experience"] - total_exp) <= 1.0) &
            (abs(df["Team Lead Experience"] - lead_exp) <= 1.0) &
            (abs(df["Project Manager Experience"] - pm_exp) <= 1.0) &
            (abs(df["Certifications"] - certs) <= 1)
        ]
        if not similar.empty:
            st.dataframe(similar[["Total Experience", "Team Lead Experience", "Project Manager Experience", "Certifications", "Salary"]].head(5))
        else:
            st.info("No similar profiles found.")
    except FileNotFoundError:
        st.warning("Salary_Data.csv not found.")

    # SHAP explanation
    st.markdown("### 🧠 SHAP Explanation")
    explainer = shap.Explainer(model)
    shap_vals = explainer(features)
    plt.figure()
    shap.plots.waterfall(shap_vals[0], show=False)
    st.pyplot(plt.gcf())

    # Salary breakdown
    st.markdown("### 🧮 Salary Breakdown")
    base = 0.60 * predicted_salary
    team_lead_bonus = 0.10 * predicted_salary
    pm_bonus = 0.15 * predicted_salary
    cert_bonus = 0.15 * predicted_salary
    breakdown_df = pd.DataFrame({
        "Component": ["Base Pay", "Team Lead Bonus", "PM Bonus", "Certification Bonus"],
        "Amount": [base, team_lead_bonus, pm_bonus, cert_bonus]
    })
    st.bar_chart(breakdown_df.set_index("Component"))
    st.dataframe(breakdown_df.style.format({"Amount": "₹{:,.2f}"}))

    # Scenario simulation
    st.markdown("### 🎛️ Scenario Simulation")
    is_big_mnc = st.checkbox("🏢 Working at a Big MNC?")
    is_metro_city = st.checkbox("🌆 Located in Metro City?")
    has_rare_skills = st.checkbox("🧠 Possesses Rare In-Demand Skills?")

    adjustment = sum([0.10 if is_big_mnc else 0, 0.05 if is_metro_city else 0, 0.07 if has_rare_skills else 0])
    if adjustment > 0:
        adj_salary = predicted_salary * (1 + adjustment)
        st.success(f"📈 Adjusted Salary Estimate: ₹{adj_salary:,.2f} (+{int(adjustment*100)}%)")

    import yagmail
    from fpdf import FPDF

    def generate_pdf(entry, filename="report.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for key, value in entry.items():
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
        pdf.output(filename)

    def send_email_with_pdf(recipient_email, pdf_file):
        sender_email = "youremail@gmail.com"
        app_password = "your-app-password"

        yag = yagmail.SMTP(sender_email, app_password)
        yag.send(
            to=recipient_email,
            subject="Your Salary Prediction Report",
            contents="Attached is your salary prediction report.",
            attachments=pdf_file
        )

    # After saving prediction:
    generate_pdf(entry, "report.pdf")

    user_email = st.text_input("Enter your email to receive PDF report")
    if st.button("📩 Email My Report"):
        if user_email:
            send_email_with_pdf(user_email, "report.pdf")
            st.success("Email sent with PDF report!")
        else:
            st.warning("Please enter your email address.")


# ------------------- BULK UPLOAD -------------------
st.subheader("📂 Bulk Salary Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    req = ["Total Experience", "Team Lead Experience", "Project Manager Experience", "Certifications"]
    if all(col in df.columns for col in req):
        preds = model.predict(df[req])
        df["Predicted Salary"] = preds
        st.dataframe(df)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions CSV", csv_data, "bulk_salary_predictions.csv", "text/csv")
    else:
        st.error("CSV must contain columns: " + ", ".join(req))

# ------------------- VISUALIZATIONS -------------------
if st.checkbox("Show salary prediction trend by Total Experience"):
    exp_range = np.arange(0, 21, 1)
    inputs = [[exp, lead_exp, pm_exp, certs] for exp in exp_range]
    salaries = model.predict(inputs)
    fig, ax = plt.subplots()
    ax.plot(exp_range, salaries, marker='o')
    ax.set_xlabel("Total Experience")
    ax.set_ylabel("Predicted Salary")
    ax.set_title("Salary vs Total Experience")
    st.pyplot(fig)

if st.checkbox("Show Feature Importance"):
    try:
        importances = model.feature_importances_
        features = ['Total Experience', 'Team Lead Experience', 'Project Manager Experience', 'Certifications']
        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance (XGBoost)")
        st.pyplot(fig)
    except:
        st.warning("Feature importance not available for this model.")

# ------------------- USER HISTORY -------------------
if st.checkbox("📜 Show My Prediction History"):
    filename = f"{st.session_state.username}_predictions.csv"
    if os.path.exists(filename):
        history_df = pd.read_csv(filename)
        st.dataframe(history_df.tail(10))
    else:
        st.info("No history available yet.")
