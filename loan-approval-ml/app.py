from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.feature_engineering import prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
METRICS_PATH = PROJECT_ROOT / "outputs" / "model_comparison.csv"


def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def get_model():
    return load_model()


def predict_with_model(model, raw_df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_dataset(raw_df)
    X = prepared.drop(columns=["Loan_ID"], errors="ignore")
    probability = model.predict_proba(X)
    classes = model.named_steps["model"].classes_
    positive_idx = 1 if 1 in classes else 0
    approval = probability[:, positive_idx]
    rejection = 1 - approval
    predicted = (approval >= 0.5).astype(int)
    final_predictions = pd.Series(predicted).map({0: "N", 1: "Y"})
    return pd.DataFrame({
        "Loan_ID": prepared["Loan_ID"].tolist(),
        "Predicted_Loan_Status": final_predictions.tolist(),
        "Approval_Probability": approval.tolist(),
        "Rejection_Probability": rejection.tolist(),
    })


def draw_feature_importance(model):
    try:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        feature_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feature_df = feature_df.sort_values("Importance", ascending=False).head(10)
        return feature_df
    except Exception:
        return pd.DataFrame({"Feature": ["N/A"], "Importance": [0.0]})


def status_badge(label: str, tone: str = "positive") -> str:
    palette = {
        "positive": ("#22c55e", "rgba(34, 197, 94, 0.14)", "#bbf7d0"),
        "warning": ("#fbbf24", "rgba(251, 191, 36, 0.15)", "#fef3c7"),
        "neutral": ("#60a5fa", "rgba(96, 165, 250, 0.14)", "#dbeafe"),
        "danger": ("#f87171", "rgba(248, 113, 113, 0.14)", "#fee2e2"),
    }
    text_color, bg_color, border_color = palette.get(tone, palette["positive"])
    return (
        f"<span style='display:inline-block; padding:0.35rem 0.7rem; border-radius:999px; "
        f"background:{bg_color}; color:{text_color}; border:1px solid {border_color}; "
        f"font-size:0.72rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; "
        f"margin-right:0.5rem; margin-bottom:0.35rem;'>"
        f"{label}</span>"
    )


st.set_page_config(page_title="Loan Approval Prediction", page_icon="💼", layout="wide")

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f5f7fb 0%, #eef4ff 100%);
            color: #0f172a;
        }
        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #edf5ff 100%);
        }
        .bank-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.9rem 1.1rem;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }
        .brand-block {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .brand-mark {
            width: 2.2rem;
            height: 2.2rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 800;
            box-shadow: 0 10px 22px rgba(59, 130, 246, 0.28);
        }
        .brand-name {
            font-weight: 800;
            letter-spacing: 0.04em;
            color: #0f172a;
        }
        .nav-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .nav-pill {
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: #eff6ff;
            border: 1px solid rgba(147, 197, 253, 0.6);
            color: #1d4ed8;
            font-size: 0.76rem;
            font-weight: 700;
        }
        .hero-box {
            background: linear-gradient(135deg, #ffffff 0%, #edf4ff 28%, #dfe9ff 100%);
            padding: 1.5rem 1.5rem 1.2rem 1.5rem;
            border-radius: 20px;
            margin-bottom: 1.3rem;
            color: #0f172a;
            border: 1px solid rgba(59, 130, 246, 0.14);
            box-shadow: 0 14px 30px rgba(96, 165, 250, 0.18), 0 0 0 1px rgba(255,255,255,0.7) inset;
            position: relative;
            overflow: hidden;
        }
        .hero-box::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, rgba(14,165,233,0.12), rgba(37,99,235,0.03), rgba(16,185,129,0.08));
            pointer-events: none;
        }
        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.05em;
            position: relative;
            z-index: 1;
        }
        .hero-sub {
            font-size: 1rem;
            color: #334155;
            margin-top: 0.45rem;
            position: relative;
            z-index: 1;
        }
        .info-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 16px;
            padding: 1rem;
            height: 100%;
            box-shadow: 0 10px 18px rgba(15, 23, 42, 0.06), 0 0 18px rgba(59,130,246,0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .info-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 14px 24px rgba(37, 99, 235, 0.12), 0 0 18px rgba(14,165,233,0.12);
        }
        .card-label {
            color: #1d4ed8;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }
        .card-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f172a;
            margin-top: 0.4rem;
        }
        .card-note {
            font-size: 0.85rem;
            color: #475569;
            margin-top: 0.25rem;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }
        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }
        .stTabs [role="tab"] {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: #475569;
            border-radius: 10px 10px 0 0;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%);
            color: #1d4ed8;
            border-color: rgba(96, 165, 250, 0.26);
        }
        .status-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.4rem 0.75rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-right: 0.5rem;
            margin-bottom: 0.35rem;
        }
        .badge-positive { background: rgba(34, 197, 94, 0.12); color: #15803d; border: 1px solid rgba(34, 197, 94, 0.2); }
        .badge-warning { background: rgba(245, 158, 11, 0.10); color: #b45309; border: 1px solid rgba(245, 158, 11, 0.2); }
        .badge-neutral { background: rgba(59, 130, 246, 0.10); color: #1d4ed8; border: 1px solid rgba(59, 130, 246, 0.18); }
        .kpi-strip {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 0.8rem 1rem;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.08);
        }
        .ticker-item {
            display: inline-block;
            padding: 0.5rem 0.8rem;
            border-radius: 12px;
            background: rgba(239,246,255,0.8);
            border: 1px solid rgba(191,219,254,0.8);
            margin-right: 0.7rem;
            margin-bottom: 0.3rem;
            font-weight: 600;
            color: #1e293b;
        }
        .ticker-label {
            color: #64748b;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            display: block;
        }
        .ticker-value {
            font-size: 1.14rem;
            color: #0f172a;
        }
        .glow-card {
            background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
            border: 1px solid rgba(191,219,254,0.9);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.9) inset, 0 10px 26px rgba(59,130,246,0.12);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="bank-header">
        <div class="brand-block">
            <div class="brand-mark">L</div>
            <div class="brand-name">LendingIQ</div>
        </div>
        <div class="nav-row">
            <div class="nav-pill">Overview</div>
            <div class="nav-pill">Risk</div>
            <div class="nav-pill">Portfolio</div>
        </div>
    </div>
    <div class="hero-box">
        <p style="margin:0; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.12em; opacity:0.9; position:relative; z-index:1;">Loan intelligence</p>
        <h1 class="hero-title">Loan Approval Prediction Dashboard</h1>
        <div class="hero-sub">Smarter underwriting decisions, transparent risk signals, and fast predictions for applicants.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Project overview")
    st.write("This app loads the trained pipeline and predicts loan approval outcomes using the same preprocessing logic as the training pipeline.")
    st.markdown("---")
    st.subheader("Quick links")
    st.markdown("- [Overview](#project-summary)\n- [Single applicant](#predict-single-applicant)\n- [Bulk CSV](#bulk-csv-prediction)\n- [Model performance](#model-performance-dashboard)\n- [Feature importance](#feature-importance)")

if not MODEL_PATH.exists():
    st.error("The trained model is not available. Please run the training pipeline first.")
    st.stop()

model = get_model()

metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()

kpi_html = """
<div class="kpi-strip">
    <div style="display:flex; flex-wrap:wrap; gap:0.6rem; align-items:center;">
        <div class="ticker-item"><span class="ticker-label">Best model</span><span class="ticker-value">{best_model}</span></div>
        <div class="ticker-item"><span class="ticker-label">Accuracy</span><span class="ticker-value">{acc}</span></div>
        <div class="ticker-item"><span class="ticker-label">F1 score</span><span class="ticker-value">{f1}</span></div>
    </div>
</div>
""".format(
    best_model=metrics.iloc[0]["Model"] if not metrics.empty else "N/A",
    acc=f"{metrics.iloc[0]['Accuracy']:.3f}" if not metrics.empty else "N/A",
    f1=f"{metrics.iloc[0]['F1']:.3f}" if not metrics.empty else "N/A",
)
st.markdown(kpi_html, unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Single applicant", "Bulk CSV", "Model performance", "Feature importance"])

with tabs[0]:
    st.markdown("### Project summary", help="Overview of the workflow and deployed ML setup")
    st.markdown(
        f"{status_badge('Production-ready', 'positive')} {status_badge('Risk-aware', 'neutral')} {status_badge('Model explainable', 'warning')}",
        unsafe_allow_html=True,
    )

    overview_col1, overview_col2, overview_col3 = st.columns(3)
    with overview_col1:
        st.markdown(
            """
            <div class="glow-card">
                <div class="card-label">Portfolio health</div>
                <div class="card-value">AI assisted</div>
                <div class="card-note">Decision support built for loan review teams.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with overview_col2:
        st.markdown(
            """
            <div class="glow-card">
                <div class="card-label">Risk signal</div>
                <div class="card-value">Explainable</div>
                <div class="card-note">Uses model features and probability scoring.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with overview_col3:
        st.markdown(
            """
            <div class="glow-card">
                <div class="card-label">Deployment</div>
                <div class="card-value">Ready</div>
                <div class="card-note">Reusable pipeline for batch and single-case use.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("The system uses a preprocessing pipeline that handles missing values, encodes categorical inputs, and engineers financial features such as total income, EMI estimate, and loan-to-income ratios.")
    st.write("The model is trained with honest validation and is saved in a reusable pipeline format for deployment and inference.")

    if not metrics.empty:
        chart_df = metrics[["Model", "Accuracy", "F1"]].copy()
        fig = px.line(
            chart_df,
            x="Model",
            y=["Accuracy", "F1"],
            markers=True,
            color_discrete_sequence=["#38bdf8", "#34d399"],
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#0f172a"},
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            legend={"orientation": "h", "y": 1.1, "x": 0},
            transition={"duration": 1000},
            hovermode="x unified",
            xaxis={"showgrid": False, "tickfont": {"color": "#475569"}},
            yaxis={"gridcolor": "#e2e8f0", "zerolinecolor": "#cbd5e1"},
        )
        st.plotly_chart(fig, width="stretch", theme="streamlit")
    else:
        st.info("Model metrics are not available yet. Train the pipeline to generate the performance chart.")

    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    quick_links = [
        ("Single applicant", "Predict a single loan case"),
        ("Bulk CSV", "Upload a batch of applicants"),
        ("Model metrics", "Review validation performance"),
        ("Feature importance", "See what drives the decision"),
    ]
    for idx, (title, text) in enumerate(quick_links):
        with [quick_col1, quick_col2, quick_col3, quick_col4][idx]:
            st.markdown(f"<div class='info-card'><div class='card-label'>{title}</div><div class='card-note'>{text}</div></div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown("### Predict single applicant")
    with st.form("single_applicant_form"):
        applicant = {
            "Gender": st.selectbox("Gender", ["Male", "Female"]),
            "Married": st.selectbox("Married", ["No", "Yes"]),
            "Dependents": st.selectbox("Dependents", ["0", "1", "2", "3+"]),
            "Education": st.selectbox("Education", ["Graduate", "Not Graduate"]),
            "Self_Employed": st.selectbox("Self_Employed", ["No", "Yes"]),
            "ApplicantIncome": st.number_input("ApplicantIncome", min_value=0, value=5000),
            "CoapplicantIncome": st.number_input("CoapplicantIncome", min_value=0, value=1000),
            "LoanAmount": st.number_input("LoanAmount", min_value=0, value=120),
            "Loan_Amount_Term": st.number_input("Loan_Amount_Term", min_value=0, value=360),
            "Credit_History": st.selectbox("Credit_History", [0.0, 1.0]),
            "Property_Area": st.selectbox("Property_Area", ["Urban", "Semiurban", "Rural"]),
        }
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([applicant])
        row["Loan_ID"] = ["APP_001"]
        result = predict_with_model(model, row)
        approval = float(result["Approval_Probability"].iloc[0])
        rejection = float(result["Rejection_Probability"].iloc[0])
        st.success(f"Predicted status: {result['Predicted_Loan_Status'].iloc[0]}")
        st.metric("Approval probability", f"{approval:.2%}")
        st.metric("Rejection probability", f"{rejection:.2%}")

with tabs[2]:
    st.markdown("### Bulk CSV prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "Loan_ID" not in df.columns:
            st.error("The CSV must include a Loan_ID column.")
        else:
            predictions = predict_with_model(model, df)
            st.dataframe(predictions, width="stretch")
            csv_data = predictions.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions", csv_data, file_name="bulk_predictions.csv", mime="text/csv")

with tabs[3]:
    st.markdown("### Model performance dashboard")
    if metrics.empty:
        st.info("Training metrics are not available yet. Run the training script to generate model_comparison.csv.")
    else:
        st.dataframe(metrics, width="stretch")

with tabs[4]:
    st.markdown("### Feature importance")
    importance_df = draw_feature_importance(model)
    st.bar_chart(importance_df.set_index("Feature")["Importance"])
    st.dataframe(importance_df, width="stretch")

st.markdown("---")
st.caption("Built for loan approval prediction, explainability, and deployment-ready ML workflows.")
