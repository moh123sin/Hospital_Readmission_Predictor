import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hospital Readmission Predictor", layout="wide", page_icon="🏥")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {background:#f0f4ff;border-radius:10px;padding:18px;text-align:center;border-left:5px solid #2E74B5;}
.risk-high   {background:#ffe0e0;border-radius:10px;padding:18px;border-left:5px solid #e74c3c;}
.risk-low    {background:#e0ffe0;border-radius:10px;padding:18px;border-left:5px solid #27ae60;}
h1 {color:#2E74B5;}
</style>""", unsafe_allow_html=True)

# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n=3000):
    np.random.seed(42)
    age          = np.random.randint(18, 90, n)
    gender       = np.random.choice(["Male","Female"], n)
    num_diagnoses= np.random.randint(1, 10, n)
    num_meds     = np.random.randint(1, 20, n)
    num_procedures= np.random.randint(0, 6, n)
    time_in_hosp = np.random.randint(1, 15, n)
    prior_admits = np.random.randint(0, 5, n)
    a1c_result   = np.random.choice(["None","Normal",">7",">8"], n, p=[0.4,0.3,0.2,0.1])
    insulin      = np.random.choice(["No","Steady","Up","Down"], n)
    discharge_to = np.random.choice(["Home","SNF","Rehab","AMA"], n, p=[0.6,0.2,0.15,0.05])

    # Readmission probability based on risk factors
    prob = (0.1
            + 0.003 * age
            + 0.05  * num_diagnoses
            + 0.02  * prior_admits
            + 0.03  * (a1c_result == ">8").astype(int)
            + 0.04  * (discharge_to == "AMA").astype(int)
            - 0.02  * (discharge_to == "Home").astype(int))
    prob = np.clip(prob, 0.05, 0.95)
    readmitted = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        "age": age, "gender": gender,
        "num_diagnoses": num_diagnoses, "num_medications": num_meds,
        "num_procedures": num_procedures, "time_in_hospital": time_in_hosp,
        "prior_admissions": prior_admits, "a1c_result": a1c_result,
        "insulin": insulin, "discharge_to": discharge_to,
        "readmitted": readmitted
    })

@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    df2 = df.copy()
    for col in ["gender","a1c_result","insulin","discharge_to"]:
        df2[col] = le.fit_transform(df2[col])

    X = df2.drop("readmitted", axis=1)
    y = df2["readmitted"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
    model.fit(X_res, y_res)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    auc     = roc_auc_score(y_test, y_proba)
    cv      = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()

    return model, X_test, y_test, y_pred, y_proba, auc, cv, X.columns.tolist()

# ── Load ──────────────────────────────────────────────────────────────────────
df = generate_data()
model, X_test, y_test, y_pred, y_proba, auc, cv, feature_names = train_model(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Hospital Readmission Predictor")
st.markdown("*Predict 30-day patient readmission risk using machine learning*")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predict Patient", "📈 Model Performance", "🔍 Data Explorer"])

# ════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    readmit_rate = df["readmitted"].mean()*100
    c1.metric("Total Patients",   f"{len(df):,}")
    c2.metric("Readmitted",       f"{df['readmitted'].sum():,}")
    c3.metric("Readmission Rate", f"{readmit_rate:.1f}%")
    c4.metric("Model AUC",        f"{auc:.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Readmission by Age Group**")
        df2 = df.copy()
        df2["age_group"] = pd.cut(df2["age"], bins=[0,30,50,65,80,100],
                                   labels=["<30","30-50","50-65","65-80","80+"])
        age_r = df2.groupby("age_group")["readmitted"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6,3.5))
        bars = ax.bar(age_r["age_group"].astype(str), age_r["readmitted"]*100,
                      color=["#2E74B5","#3A8FD4","#5BA3E0","#7BBDED","#9ED0F5"])
        ax.set_ylabel("Readmission Rate (%)")
        ax.set_xlabel("Age Group")
        ax.set_title("Readmission Rate by Age Group")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                    f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Readmission by Discharge Destination**")
        dis_r = df.groupby("discharge_to")["readmitted"].mean().reset_index().sort_values("readmitted", ascending=True)
        fig, ax = plt.subplots(figsize=(6,3.5))
        colors = ["#27ae60","#2E74B5","#e67e22","#e74c3c"]
        ax.barh(dis_r["discharge_to"], dis_r["readmitted"]*100, color=colors)
        ax.set_xlabel("Readmission Rate (%)")
        ax.set_title("Readmission Rate by Discharge Type")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Feature Importance**")
        fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(8)
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.barh(fi.index, fi.values, color="#2E74B5")
        ax.set_title("Top Predictors of Readmission")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("**Prior Admissions vs Readmission Rate**")
        pr = df.groupby("prior_admissions")["readmitted"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.plot(pr["prior_admissions"], pr["readmitted"]*100, marker="o", color="#2E74B5", linewidth=2)
        ax.fill_between(pr["prior_admissions"], pr["readmitted"]*100, alpha=0.15, color="#2E74B5")
        ax.set_xlabel("Prior Admissions")
        ax.set_ylabel("Readmission Rate (%)")
        ax.set_title("Readmission Risk vs Prior History")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════
# TAB 2 — Predict Patient
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔮 Predict Readmission Risk for a New Patient")
    st.markdown("Fill in the patient details below:")

    c1, c2, c3 = st.columns(3)
    with c1:
        age           = st.slider("Age", 18, 90, 55)
        gender        = st.selectbox("Gender", ["Male","Female"])
        num_diagnoses = st.slider("Number of Diagnoses", 1, 10, 4)
    with c2:
        num_meds      = st.slider("Number of Medications", 1, 20, 8)
        num_procedures= st.slider("Number of Procedures", 0, 6, 1)
        time_in_hosp  = st.slider("Days in Hospital", 1, 15, 4)
    with c3:
        prior_admits  = st.slider("Prior Admissions", 0, 5, 1)
        a1c_result    = st.selectbox("HbA1c Result", ["None","Normal",">7",">8"])
        insulin       = st.selectbox("Insulin", ["No","Steady","Up","Down"])
        discharge_to  = st.selectbox("Discharge To", ["Home","SNF","Rehab","AMA"])

    if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
        enc_map = {
            "gender":       {"Male":1,"Female":0},
            "a1c_result":   {"None":1,"Normal":2,">7":0,">8":3},
            "insulin":      {"Down":0,"No":1,"Steady":2,"Up":3},
            "discharge_to": {"AMA":0,"Home":1,"Rehab":2,"SNF":3},
        }
        input_data = pd.DataFrame([[
            age,
            enc_map["gender"][gender],
            num_diagnoses, num_meds, num_procedures, time_in_hosp, prior_admits,
            enc_map["a1c_result"][a1c_result],
            enc_map["insulin"][insulin],
            enc_map["discharge_to"][discharge_to],
        ]], columns=feature_names)

        prob = model.predict_proba(input_data)[0][1]
        risk = "HIGH" if prob >= 0.5 else "LOW"

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Readmission Probability", f"{prob*100:.1f}%")
        r2.metric("Risk Level", risk)
        r3.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")

        if risk == "HIGH":
            st.markdown(f"""<div class='risk-high'>
            <h3>⚠️ HIGH Readmission Risk ({prob*100:.1f}%)</h3>
            <p>This patient has elevated readmission risk. Consider: enhanced discharge planning,
            follow-up within 7 days, medication reconciliation, and care coordination.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='risk-low'>
            <h3>✅ LOW Readmission Risk ({prob*100:.1f}%)</h3>
            <p>Standard discharge protocols are appropriate. Schedule routine 30-day follow-up.</p>
            </div>""", unsafe_allow_html=True)

        # Risk gauge
        fig, ax = plt.subplots(figsize=(5,1.5))
        ax.barh(["Risk"], [prob], color="#e74c3c" if risk=="HIGH" else "#27ae60", height=0.4)
        ax.barh(["Risk"], [1-prob], left=[prob], color="#eee", height=0.4)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0,1)
        ax.set_title(f"Risk Score: {prob*100:.1f}%")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Model Performance Metrics")

    m1,m2,m3,m4 = st.columns(4)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    m1.metric("ROC-AUC",   f"{auc:.4f}")
    m2.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.4f}")
    m3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    m4.metric("Recall",    f"{recall_score(y_test, y_pred):.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(fpr, tpr, color="#2E74B5", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0,1],[0,1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not Readmitted","Readmitted"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Classification Report**")
    report = classification_report(y_test, y_pred,
                                   target_names=["Not Readmitted","Readmitted"],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 4 — Data Explorer
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔍 Explore the Dataset")
    col1, col2 = st.columns(2)
    with col1:
        age_filter = st.slider("Filter by Age Range", 18, 90, (18, 90))
    with col2:
        gender_filter = st.multiselect("Filter by Gender", ["Male","Female"], default=["Male","Female"])

    filtered = df[(df["age"].between(*age_filter)) & (df["gender"].isin(gender_filter))]
    st.markdown(f"**Showing {len(filtered):,} patients**")
    st.dataframe(filtered.head(100), use_container_width=True)

    st.download_button("📥 Download Dataset as CSV",
                       data=df.to_csv(index=False),
                       file_name="hospital_readmission_data.csv",
                       mime="text/csv")
