from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from churn.features import align_model_numeric_types, prepare_scoring_frame
from churn.retention import add_retention_recommendations

st.set_page_config(page_title="Churn Risk Studio", page_icon="📉", layout="wide")
st.title("Customer Churn Risk Studio")
st.caption("Upload customer records to prioritize retention outreach.")
model_path=Path("models/churn_model.joblib")
if not model_path.exists(): st.warning("Train a model first: `python -m churn.train`"); st.stop()
artifact=joblib.load(model_path)
uploaded=st.file_uploader("Telco customer CSV", type="csv")
if uploaded:
    try:
        raw=pd.read_csv(uploaded); X=prepare_scoring_frame(raw).reindex(columns=artifact.feature_columns)
        X=align_model_numeric_types(X,artifact.pipeline); p=artifact.pipeline.predict_proba(X)[:,1]
    except (TypeError, ValueError) as exc:
        st.error(f"Could not score this CSV. Check that it uses the expected Telco customer columns and values. Details: {exc}")
        st.stop()
    out=raw.copy(); out["churn_probability"]=p; out["risk_band"]=pd.cut(p,[-1,.3,.6,1],labels=["Low","Medium","High"])
    out=add_retention_recommendations(out,p)
    a,b,c=st.columns(3); a.metric("Customers",len(out)); b.metric("High risk",int((p>=.6).sum())); c.metric("Average risk",f"{p.mean():.1%}")
    st.bar_chart(out["risk_band"].value_counts())
    st.subheader("Staff retention action list")
    st.caption("Recommendations are decision support. Confirm eligibility and customer consent before acting.")
    display_columns=[c for c in ["customerID","churn_probability","risk_band","staff_priority","recommended_action","recommendation_reason"] if c in out]
    st.dataframe(out.sort_values("churn_probability",ascending=False)[display_columns],use_container_width=True)
    st.download_button("Download scored customers",out.to_csv(index=False),"churn_predictions.csv","text/csv")
