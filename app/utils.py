import pandas as pd

def preprocess_inputs_for_explanation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure numeric fields are in numeric format for rule-based explanations.
    """
    df = data.copy()

    # Convert Dependents "3+" to 3, etc.
    if "Dependents" in df.columns:
        dep = df.at[0, "Dependents"]
        if isinstance(dep, str) and dep.strip() == "3+":
            df.at[0, "Dependents"] = 3
        else:
            try:
                df.at[0, "Dependents"] = int(dep)
            except:
                df.at[0, "Dependents"] = 0

    # Convert types safely
    numeric_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]
    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    return df


def generate_natural_explanation(row: pd.Series, prediction: int) -> str:
    """
    Simple rule-based explanation to describe WHY the model may have
    approved / rejected the loan, in human language.
    """

    income = row.get("ApplicantIncome", 0)
    co_income = row.get("CoapplicantIncome", 0)
    total_income = income + co_income
    loan_amount = row.get("LoanAmount", 0)
    term = row.get("Loan_Amount_Term", 360)
    credit = row.get("Credit_History", 0)
    dependents = row.get("Dependents", 0)
    education = row.get("Education", "Graduate")
    employed = row.get("Self_Employed", "No")
    area = row.get("Property_Area", "Urban")

    reasons = []

    # Income vs loan (rough EMI)
    if total_income > 0 and loan_amount > 0:
        emi_estimate = loan_amount / max(term, 1)
        income_ratio = emi_estimate / max(total_income, 1)
        if income_ratio < 0.2:
            reasons.append("your combined income is strong compared to the EMI burden")
        elif income_ratio < 0.4:
            reasons.append("your income is just sufficient for the EMI")
        else:
            reasons.append("the EMI looks high compared to your income")

    # Credit history
    if credit == 1:
        reasons.append("you have a good credit history")
    else:
        reasons.append("your credit history is either missing or not strong")

    # Dependents
    try:
        dep_num = int(dependents)
    except:
        dep_num = 0

    if dep_num >= 3:
        reasons.append("you have higher family dependents, which increases financial load")
    else:
        reasons.append("you have fewer dependents, which reduces financial risk")

    # Employment
    if str(employed).strip().lower() == "no":
        reasons.append("you are a salaried/non-self-employed applicant, which is often more stable")
    else:
        reasons.append("being self-employed may introduce some income variability")

    # Area
    reasons.append(f"your property is in a {area.lower()} area, which also influences risk and collateral value")

    # Education
    if education == "Graduate":
        reasons.append("your graduate education can be associated with better earning potential")
    else:
        reasons.append("non-graduate education may slightly affect perceived earning stability")

    # Build final explanation text
    if prediction == 1:
        decision_text = "Your loan was **approved**."
        tone = "The main reasons are that"
    else:
        decision_text = "Your loan was **rejected**."
        tone = "The likely reasons are that"

    reasons_text = "; ".join(reasons[:4])

    explanation = (
        f"{decision_text} {tone} {reasons_text}. "
        "This is a simplified explanation to help you understand the decision in human terms."
    )

    return explanation
