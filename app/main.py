import streamlit as st
import pickle
import pandas as pd
import os
import json
import datetime as dt
import matplotlib.pyplot as plt

from utils import preprocess_inputs_for_explanation, generate_natural_explanation

# ---------------- PATHS / GLOBALS ----------------
log_file = os.path.join("logs", "audit_log.json")

# ---------------- BASIC CONFIG ----------------
st.set_page_config(page_title="Ethical AI Loan Approval System", layout="centered")
st.title("🏦 Ethical AI Loan Approval System")
st.caption("Theme: Ethical AI in Banking – Building Trust & Transparency")

# ---------------- LOAD MODEL ----------------
model_path = os.path.join("models", "loan_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# ---------------- SIDEBAR: DATA CONSENT ----------------
st.sidebar.title("🔐 Data Consent & Control")

st.sidebar.write("Choose which data fields the AI is allowed to use for decision-making:")

consent = {}
consent_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Dependents",
]

for feature in consent_features:
    consent[feature] = st.sidebar.checkbox(f"Allow {feature}", value=True)

st.sidebar.info(
    "If a feature is disabled, its value will be neutralized before prediction.\n"
    "This demonstrates customer control over personal data."
)

# ---------------- MAIN FORM ----------------
st.subheader("Enter Your Details to Check Loan Eligibility")

with st.form("loan_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married Status", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0.0, step=1000.0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0, step=1000.0)
    LoanAmount = st.number_input("Loan Amount", min_value=0.0, step=10000.0)
    Loan_Amount_Term = st.number_input("Loan Term (in days)", min_value=0.0, step=12.0)
    Credit_History = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1, 0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.markdown("**📄 Document Verification (Demo)**")
    Doc_ApplicantIncome = st.number_input(
        "Income as per salary slip / bank statement (for demo)",
        min_value=0.0,
        step=1000.0,
        help="In a real system, this would be extracted automatically from the uploaded document using OCR.",
    )
    uploaded_doc = st.file_uploader(
        "Upload income document (optional – PDF/Image, demo only)",
        type=["pdf", "png", "jpg", "jpeg"],
    )

    submitted = st.form_submit_button("🔍 Predict Loan Status")

# ---------------- LOAN OFFICER OVERRIDE (WIDGETS) ----------------
st.markdown("---")
st.subheader("👨‍💼 Loan Officer Review & Override (Optional)")

st.write(
    "This section is intended for authorised bank staff. "
    "They can confirm the AI recommendation or override it with a clear justification."
)

override_mode = st.radio(
    "Final decision by officer:",
    ["Use AI recommendation", "Force Approve", "Force Reject"],
)

officer_name = st.text_input("Officer name / ID (for audit)", value="")

override_reason = st.text_area(
    "If overriding, provide justification (required for compliance):",
    height=100,
    placeholder="Example: Customer has additional income not captured in the system...",
)

# ---------------- PREDICTION SECTION ----------------
if submitted:
    # Build input dataframe
    input_df = pd.DataFrame(
        [
            [
                Gender,
                Married,
                Dependents,
                Education,
                Self_Employed,
                ApplicantIncome,
                CoapplicantIncome,
                LoanAmount,
                Loan_Amount_Term,
                Credit_History,
                Property_Area,
            ]
        ],
        columns=[
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
        ],
    )

    # Apply data consent: neutralize blocked features
    for feature, allowed in consent.items():
        if feature in input_df.columns and not allowed:
            input_df[feature] = 0  # neutralize

    # Model prediction & probability (AI recommendation)
    proba = float(model.predict_proba(input_df)[0][1])  # probability of approval
    ai_prediction = int(model.predict(input_df)[0])

    # ---------------- APPLY HUMAN OVERRIDE ----------------
    final_decision = ai_prediction
    override_used = False

    if override_mode == "Force Approve":
        final_decision = 1
        override_used = True
    elif override_mode == "Force Reject":
        final_decision = 0
        override_used = True

    if override_used and not override_reason:
        st.warning(
            "You selected a manual override but did not provide a justification. "
            "In a real system, this would not be allowed."
        )

    # ---------------- SHOW DECISION ----------------
    st.markdown("## ✅ Decision Overview")

    # AI recommendation
    if ai_prediction == 1:
        st.info(f"🤖 AI Recommendation: **Approve** (Approval probability: {proba:.2%})")
    else:
        st.info(f"🤖 AI Recommendation: **Reject** (Approval probability: {proba:.2%})")

    # Final decision after human review
    if final_decision == 1:
        st.success("🧑‍💼 Final Decision after human review: **Approved**")
    else:
        st.error("🧑‍💼 Final Decision after human review: **Rejected**")

    if override_used:
        st.warning(
            f"Human override applied by officer `{officer_name or 'Unknown'}`.\n\n"
            f"Reason: {override_reason or '(no reason provided)'}"
        )
    else:
        st.caption("No human override applied. Final decision matches the AI recommendation.")

    # ---------------- DOCUMENT CONSISTENCY & TRUST SCORE ----------------
    st.markdown("---")
    st.subheader("🛡 Document Consistency & Trust Score (Demo)")

    trust_score = None
    if Doc_ApplicantIncome and Doc_ApplicantIncome > 0:
        # Compare declared vs document income
        max_income = max(ApplicantIncome, Doc_ApplicantIncome, 1.0)
        diff = abs(ApplicantIncome - Doc_ApplicantIncome)
        diff_ratio = diff / max_income  # 0 → perfect match, 1 → big mismatch

        # Simple trust score (0–100)
        trust_score = max(0.0, min(100.0, 100 * (1 - diff_ratio)))

        if diff_ratio <= 0.1:
            alignment = "High alignment between declared income and document."
            level = "High"
        elif diff_ratio <= 0.3:
            alignment = "Some differences between declared income and document."
            level = "Medium"
        else:
            alignment = "Large mismatch between declared income and document."
            level = "Low"

        st.write(
            f"- Declared income: **{ApplicantIncome:.0f}**\n"
            f"- Document income: **{Doc_ApplicantIncome:.0f}**\n"
            f"- Approx. difference: **{diff_ratio * 100:.1f}%**\n"
            f"- **Trust score:** `{trust_score:.0f} / 100` ({level} trust)\n"
        )

        st.info(
            alignment
            + " In a production system, this check would be automated using OCR "
              "and backend verification against official statements."
        )
    else:
        st.info(
            "No document income provided. Trust score cannot be computed. "
            "In a real bank, document verification would be mandatory for larger loans."
        )

    # ---------------- WHAT-IF LOAN OPTIMISATION ----------------
    st.markdown("---")
    st.subheader("🔄 What-If Loan Optimisation (AI Suggestions)")

    scenarios = []
    base_df = input_df.copy()

    # Scenario 1: Reduce loan amount by 20%
    if LoanAmount > 0:
        s1_df = base_df.copy()
        s1_df["LoanAmount"] = LoanAmount * 0.8
        proba_s1 = float(model.predict_proba(s1_df)[0][1])
        scenarios.append(
            {
                "Scenario": "Reduce loan amount by 20%",
                "LoanAmount": s1_df["LoanAmount"].iloc[0],
                "Loan_Amount_Term": s1_df["Loan_Amount_Term"].iloc[0],
                "Approval Probability": f"{proba_s1:.1%}",
            }
        )

    # Scenario 2: Increase loan term by 25%
    if Loan_Amount_Term > 0:
        s2_df = base_df.copy()
        s2_df["Loan_Amount_Term"] = Loan_Amount_Term * 1.25
        proba_s2 = float(model.predict_proba(s2_df)[0][1])
        scenarios.append(
            {
                "Scenario": "Increase loan term by 25%",
                "LoanAmount": s2_df["LoanAmount"].iloc[0],
                "Loan_Amount_Term": s2_df["Loan_Amount_Term"].iloc[0],
                "Approval Probability": f"{proba_s2:.1%}",
            }
        )

    # Scenario 3: Reduce loan amount 20% + increase term 25%
    if LoanAmount > 0 and Loan_Amount_Term > 0:
        s3_df = base_df.copy()
        s3_df["LoanAmount"] = LoanAmount * 0.8
        s3_df["Loan_Amount_Term"] = Loan_Amount_Term * 1.25
        proba_s3 = float(model.predict_proba(s3_df)[0][1])
        scenarios.append(
            {
                "Scenario": "Reduce loan by 20% & increase term by 25%",
                "LoanAmount": s3_df["LoanAmount"].iloc[0],
                "Loan_Amount_Term": s3_df["Loan_Amount_Term"].iloc[0],
                "Approval Probability": f"{proba_s3:.1%}",
            }
        )

    if scenarios:
        scenario_df = pd.DataFrame(scenarios)
        st.write("**How you could improve your approval chances or reduce EMI:**")
        st.dataframe(scenario_df, use_container_width=True)

        # Simple textual recommendation – pick highest probability
        try:
            # Extract numeric probability again
            probs = []
            for s in scenarios:
                p = float(s["Approval Probability"].strip("%")) / 100.0
                probs.append(p)
            best_idx = int(pd.Series(probs).idxmax())
            best_scenario = scenarios[best_idx]["Scenario"]
            st.success(
                f"Recommendation: **{best_scenario}** gives the best approval outlook among these options."
            )
        except Exception:
            pass
    else:
        st.info("What-if suggestions are not available for this combination of inputs.")

    # ---------------- NATURAL LANGUAGE EXPLANATION ----------------
    st.markdown("---")
    st.subheader("🧠 Why did the AI make this decision?")

    clean_df = preprocess_inputs_for_explanation(input_df)
    row = clean_df.iloc[0]
    explanation = generate_natural_explanation(row, ai_prediction)
    st.write(explanation)

    # ---------------- EXPLAIN MY PROFILE (CUSTOMER VIEW) ----------------
    st.markdown("---")
    st.subheader("🧾 Explain My Profile")

    # Income level
    if ApplicantIncome < 3000:
        income_level = "low"
    elif ApplicantIncome < 8000:
        income_level = "moderate"
    else:
        income_level = "high"

    # Loan burden relative to income (rough)
    monthly_loan = LoanAmount / max(Loan_Amount_Term / 30.0, 1) if Loan_Amount_Term > 0 else 0
    if ApplicantIncome == 0:
        burden_text = "cannot be computed because income is 0."
        stress_level = "Unknown"
        wellness_score = None
    else:
        ratio = monthly_loan / ApplicantIncome
        if ratio < 0.3:
            burden_text = "your EMI compared to income looks comfortable."
            stress_level = "Low"
        elif ratio < 0.6:
            burden_text = "your EMI compared to income is moderate and may be acceptable."
            stress_level = "Medium"
        else:
            burden_text = "your EMI compared to income is high and may increase risk."
            stress_level = "High"

        # Simple financial wellness score (higher is better)
        wellness_score = max(0.0, min(100.0, 100 * (1 - ratio)))

    credit_text = "good" if Credit_History == 1 else "poor or unknown"
    area_text = (
        "an urban area"
        if Property_Area == "Urban"
        else "a semi-urban area"
        if Property_Area == "Semiurban"
        else "a rural area"
    )

    st.write(
        f"- **Income level the AI sees:** `{income_level}` (around {ApplicantIncome:.0f} per month)\n"
        f"- **Credit history according to records:** `{credit_text}`\n"
        f"- **Property area:** `{Property_Area}` (interpreted as {area_text})\n"
        f"- **Loan burden view:** based on your loan amount and term, {burden_text}\n"
    )

    if wellness_score is not None:
        st.write(
            f"- **Financial wellness indicator:** `{wellness_score:.0f} / 100` "
            f"(stress level: **{stress_level}**)"
        )

    st.info(
        "This section summarises how the AI currently understands your profile. "
        "If any of this information is incorrect, you should be able to request a correction."
    )

    if st.button("🛠 Request correction to my profile"):
        st.success(
            "Your correction request has been recorded. "
            "In a real banking system, this would notify a human officer to review and update your data."
        )

    # ---------------- FEATURE IMPORTANCE VISUAL (TRANSPARENCY) ----------------
    st.markdown("---")
    st.subheader("📊 Feature Importance – How the Model Thinks (Global View)")

    try:
        rf_model = model.named_steps["model"]
        preprocess = model.named_steps["preprocess"]

        feature_names = preprocess.get_feature_names_out()
        importances = rf_model.feature_importances_

        fi_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values(by="importance", ascending=False)
            .head(10)
        )

        st.write("Top 10 most important model features:")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(fi_df["feature"], fi_df["importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Global Feature Importance (Random Forest)")
        st.pyplot(fig)

        st.info(
            "This chart shows which features the model generally relies on most. "
            "It supports transparency and helps auditors understand model behavior."
        )
    except Exception as e:
        st.warning(f"Could not display feature importance due to: {e}")

    # ---------------- AUDIT LOGGING (GOVERNANCE) ----------------
    log_entry = {
        "timestamp": str(dt.datetime.now()),
        "input": input_df.to_dict(),
        "prediction": ai_prediction,                 # raw AI prediction
        "final_decision": int(final_decision),       # after human review
        "approval_probability": float(proba),
        "consent_used": consent,
        "override_used": override_used,
        "override_mode": override_mode,
        "override_reason": override_reason,
        "officer_name": officer_name,
        "doc_income": float(Doc_ApplicantIncome),
        "doc_uploaded": uploaded_doc is not None,
        "trust_score": float(trust_score) if trust_score is not None else None,
        "financial_wellness": float(wellness_score) if 'wellness_score' in locals() and wellness_score is not None else None,
    }

    os.makedirs("logs", exist_ok=True)

    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                content = f.read().strip()
                if content:
                    logs = json.loads(content)
                    if isinstance(logs, dict):
                        logs = [logs]
        except json.JSONDecodeError:
            logs = []

    logs.append(log_entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)

    st.caption("This decision has been recorded in the audit log for transparency and oversight.")

# ---------------- GOVERNANCE & FAIRNESS DASHBOARD (AUDITOR VIEW) ----------------
st.markdown("---")
st.header("🛡️ AI Governance & Fairness Dashboard (Auditor View)")

gov_logs = []
if os.path.exists(log_file):
    try:
        with open(log_file, "r") as f:
            content = f.read().strip()
            if content:
                gov_logs = json.loads(content)
                if isinstance(gov_logs, dict):
                    gov_logs = [gov_logs]
    except json.JSONDecodeError:
        gov_logs = []

if not gov_logs:
    st.info("No audit log entries yet. Submit a few loan applications to populate the dashboard.")
else:
    # Flatten logs into DataFrame
    records = []
    for entry in gov_logs:
        flat_input = {}
        for k, v in entry.get("input", {}).items():
            if isinstance(v, dict):
                flat_input[k] = list(v.values())[0]
            else:
                flat_input[k] = v

        records.append(
            {
                "timestamp": entry.get("timestamp"),
                "prediction": entry.get("prediction"),
                "final_decision": entry.get("final_decision"),
                "approval_probability": entry.get("approval_probability"),
                "officer_name": entry.get("officer_name"),
                "override_used": entry.get("override_used"),
                "override_mode": entry.get("override_mode"),
                "trust_score": entry.get("trust_score"),
                "financial_wellness": entry.get("financial_wellness"),
                **flat_input,
            }
        )

    log_df = pd.DataFrame(records)

    st.subheader("📜 Audit Log (Recent Decisions)")
    st.dataframe(
        log_df.sort_values("timestamp", ascending=False).head(50),
        use_container_width=True,
    )

    # ---------------- Fairness / Bias Checks ----------------
    st.subheader("⚖️ Fairness & Bias Monitoring")

    gender_stats = None
    area_stats = None

    if "Gender" in log_df.columns:
        gender_stats = log_df.groupby("Gender")["prediction"].mean().mul(100).round(1)
        st.markdown("**Approval Rate by Gender (AI model)**")
        st.bar_chart(gender_stats)
        st.write(gender_stats.to_frame("Approval rate (%)"))

    if "Property_Area" in log_df.columns:
        area_stats = log_df.groupby("Property_Area")["prediction"].mean().mul(100).round(1)
        st.markdown("**Approval Rate by Property Area (AI model)**")
        st.bar_chart(area_stats)
        st.write(area_stats.to_frame("Approval rate (%)"))

    try:
        if gender_stats is not None and gender_stats.max() - gender_stats.min() >= 15:
            st.warning(
                "⚠️ There is a noticeable gap (≥ 15%) in approval rates between genders. "
                "This may indicate potential bias and should be reviewed by human auditors."
            )

        if area_stats is not None and area_stats.max() - area_stats.min() >= 15:
            st.warning(
                "⚠️ There is a noticeable gap (≥ 15%) in approval rates between property areas. "
                "This may indicate potential geographic or socio-economic bias."
            )
    except Exception:
        pass

    # ---------------- RISK & OPERATIONS ALERTS ----------------
    st.subheader("🚨 Risk & Operations Alerts")

    # Overall approval rate (final decision)
    if "final_decision" in log_df.columns:
        approval_rate_final = float(log_df["final_decision"].mean() * 100)
        st.write(f"- Overall final approval rate: **{approval_rate_final:.1f}%**")

        if approval_rate_final <= 20:
            st.warning(
                "Very low overall approval rate. This may indicate an overly conservative policy "
                "or a model that is rejecting too many customers."
            )
        elif approval_rate_final >= 90:
            st.warning(
                "Very high overall approval rate. This may increase default risk and should be "
                "reviewed by risk management."
            )

    # Override rate
    if "override_used" in log_df.columns:
        override_rate = float(log_df["override_used"].mean() * 100)
        st.write(f"- Human override usage rate: **{override_rate:.1f}%**")

        if override_rate >= 30:
            st.warning(
                "High level of human overrides. This may indicate that the AI recommendations "
                "are not fully aligned with policy or frontline experience."
            )

    # Average trust score
    if "trust_score" in log_df.columns:
        avg_trust = log_df["trust_score"].dropna()
        if not avg_trust.empty:
            avg_trust_val = float(avg_trust.mean())
            st.write(f"- Average document trust score: **{avg_trust_val:.1f} / 100**")
            if avg_trust_val < 60:
                st.warning(
                    "Average trust score is low. Document inconsistencies may be common and "
                    "deserve further investigation."
                )

    # Duplicate / repeated applications (simple heuristic)
    duplicate_cols = [c for c in ["Gender", "ApplicantIncome", "LoanAmount", "Property_Area"] if c in log_df.columns]
    if duplicate_cols:
        dup_mask = log_df.duplicated(subset=duplicate_cols, keep=False)
        dup_count = int(dup_mask.sum())
        if dup_count > 0:
            st.warning(
                f"Detected **{dup_count}** potential repeated/duplicate applications "
                f"based on {', '.join(duplicate_cols)}. This may indicate gaming or fraud attempts."
            )
