"""Transparent, rule-based retention guidance for scored customers."""

from __future__ import annotations

import pandas as pd


def _value(row: pd.Series, column: str, default=None):
    value = row.get(column, default)
    return default if pd.isna(value) else value


def recommend_action(row: pd.Series, churn_probability: float) -> tuple[str, str]:
    """Return a staff action and reason without using demographic attributes."""
    if churn_probability < 0.30:
        return (
            "Maintain regular service and monitor at the next review.",
            "Customer is currently in the low-risk band.",
        )

    actions: list[str] = []
    reasons: list[str] = []

    if _value(row, "Contract") == "Month-to-month":
        actions.append("Discuss an eligible longer-term plan or loyalty offer")
        reasons.append("month-to-month contract")
    if _value(row, "TechSupport") == "No":
        actions.append("Offer a proactive technical-support check-in")
        reasons.append("no technical-support service")
    if _value(row, "OnlineSecurity") == "No":
        actions.append("Explain eligible online-security or support options")
        reasons.append("no online-security service")
    if _value(row, "PaymentMethod") == "Electronic check":
        actions.append("Offer help switching to a convenient automatic payment method")
        reasons.append("electronic-check payment")
    if float(_value(row, "tenure", 999)) <= 6:
        actions.append("Schedule a new-customer onboarding call")
        reasons.append("tenure of six months or less")

    if not actions:
        actions.append("Contact the customer to understand service needs and select an eligible offer")
        reasons.append("elevated predicted churn risk")

    # Keep the dashboard concise while still giving staff more than one useful option.
    return "; ".join(actions[:3]) + ".", ", ".join(reasons[:3]).capitalize() + "."


def add_retention_recommendations(
    customers: pd.DataFrame, probabilities
) -> pd.DataFrame:
    """Append review priority, recommended actions, and their reasons."""
    result = customers.copy()
    probabilities = pd.Series(probabilities, index=result.index, dtype=float)
    recommendations = [
        recommend_action(row, probabilities.loc[index])
        for index, row in result.iterrows()
    ]
    result["staff_priority"] = pd.cut(
        probabilities, [-1, 0.30, 0.60, 1], labels=["Routine", "Review", "Urgent"]
    )
    result["recommended_action"] = [action for action, _ in recommendations]
    result["recommendation_reason"] = [reason for _, reason in recommendations]
    return result
