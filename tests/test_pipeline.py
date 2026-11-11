import pandas as pd
from churn.data import validate
from churn.features import align_model_numeric_types, prepare_frame, prepare_scoring_frame
from churn.model import build_pipeline, candidates, choose_threshold
from churn.retention import add_retention_recommendations, recommend_action

def sample():
    rows=[]
    for i in range(16):
        rows.append({"customerID":f"C{i}","gender":"Male" if i%2 else "Female","SeniorCitizen":i%2,"Partner":"Yes","Dependents":"No","tenure":i+1,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"Yes" if i%3 else "No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month" if i%2 else "Two year","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":30+i,"TotalCharges":str((30+i)*(i+1)),"Churn":"Yes" if i%3==0 else "No"})
    return pd.DataFrame(rows)

def test_features_and_training():
    df=sample(); validate(df); X,y=prepare_frame(df)
    assert {"AvgMonthlySpend","IsNewCustomer","ServiceCount"}.issubset(X.columns)
    pipe=build_pipeline(X,candidates()["logistic_regression"]); pipe.fit(X,y)
    assert pipe.predict_proba(X).shape==(len(X),2)

def test_scoring_and_threshold():
    df=sample(); X=prepare_scoring_frame(df.drop(columns="Churn"))
    assert "customerID" not in X and 0 < choose_threshold([0,0,1,1],[.1,.4,.6,.9]) < 1

def test_scoring_normalizes_text_numeric_columns():
    df=sample().drop(columns="Churn").iloc[:2].copy()
    df["SeniorCitizen"]=["No","Yes"]
    df["tenure"]=["1","2"]
    df["MonthlyCharges"]=["30.5","bad value"]
    X=prepare_scoring_frame(df)
    assert X["SeniorCitizen"].tolist() == [0,1]
    assert X["tenure"].tolist() == [1,2]
    assert X["MonthlyCharges"].isna().sum() == 1

def test_scoring_aligns_columns_with_fitted_numeric_transformer():
    df=sample(); X,y=prepare_frame(df)
    pipe=build_pipeline(X,candidates()["logistic_regression"]); pipe.fit(X,y)
    scoring=prepare_scoring_frame(df.drop(columns="Churn"))
    scoring["ServiceCount"]="No"
    aligned=align_model_numeric_types(scoring,pipe)
    assert aligned["ServiceCount"].isna().all()
    assert pipe.predict_proba(aligned).shape == (len(aligned),2)

def test_retention_recommendations_are_actionable_and_risk_aware():
    df=sample().drop(columns="Churn").iloc[:2].copy()
    df.loc[df.index[0], ["Contract","TechSupport","tenure"]] = ["Month-to-month","No",2]
    result=add_retention_recommendations(df,[.85,.10])
    assert result.loc[result.index[0],"staff_priority"] == "Urgent"
    assert "longer-term plan" in result.loc[result.index[0],"recommended_action"]
    assert result.loc[result.index[1],"staff_priority"] == "Routine"
    assert "monitor" in result.loc[result.index[1],"recommended_action"]

def test_recommendations_handle_missing_optional_fields():
    action, reason=recommend_action(pd.Series({"customerID":"C1"}),.70)
    assert "Contact the customer" in action
    assert "churn risk" in reason
