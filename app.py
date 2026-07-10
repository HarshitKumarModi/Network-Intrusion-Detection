import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.pipeline.predictor import predict
from src.comparison.compare_models import compare_models


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.metric-card{
    background:#1f2937;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
    box-shadow:0px 0px 12px rgba(0,0,0,0.35);
}

.metric-title{
    font-size:18px;
    color:#b0b0b0;
}

.metric-value{
    font-size:34px;
    font-weight:bold;
    color:#00E5FF;
}

.big-title{
    font-size:48px;
    font-weight:700;
    color:#4FC3F7;
}

.sub-title{
    font-size:22px;
    color:white;
}

.footer{
    text-align:center;
    color:gray;
    padding:20px;
}

hr{
    border:1px solid #333;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3064/3064197.png",
    width=90
)

st.sidebar.title("Dashboard")

st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(
    "Choose ML Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "Artificial Neural Network",
        "XGBoost"
    ]
)

st.sidebar.success(f"Selected Model\n\n**{selected_model}**")

st.sidebar.markdown("---")

st.sidebar.subheader("Available Models")

st.sidebar.markdown("""
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- Artificial Neural Network
- XGBoost
""")

st.sidebar.markdown("---")

st.sidebar.subheader("Developer")

st.sidebar.write("Harshit Kumar Modi")

st.sidebar.write("B.Tech CSE")

st.sidebar.write("VIT Bhopal University")


# =====================================================
# HEADER
# =====================================================

st.markdown(
    "<h1 class='big-title'>🛡️ Network Intrusion Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>AI Powered Cyber Security Dashboard</p>",
    unsafe_allow_html=True
)

st.write(
    "Detect malicious network traffic using Machine Learning."
)

st.markdown("---")

# =====================================================
# UPLOAD DATASET
# =====================================================

st.header("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload Processed CSV File",
    type=["csv"]
)

# =====================================================
# IF FILE IS UPLOADED
# =====================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset uploaded successfully!")

    st.markdown("---")

    # ==========================================
    # DATASET OVERVIEW
    # ==========================================

    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Selected Model",
            selected_model
        )

    st.markdown("---")

    # ==========================================
    # DATASET PREVIEW
    # ==========================================

    st.subheader("👀 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================
    # DATASET INFORMATION
    # ==========================================

    with st.expander("📋 Dataset Information"):

        info_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str)
        })

        st.dataframe(
            info_df,
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================
    # RUN PREDICTION BUTTON
    # ==========================================

    predict_btn = st.button(
        "🚀 Run Intrusion Detection",
        use_container_width=True
    )

        # =====================================================
    # RUN PREDICTION
    # =====================================================

    if predict_btn:

        with st.spinner("Running Intrusion Detection..."):

            predictions = predict(selected_model, df)

        st.success("✅ Prediction Completed Successfully!")

        result_df = df.copy()

        result_df["Prediction"] = predictions

        st.markdown("---")

        # =====================================================
        # PREDICTION RESULTS
        # =====================================================

        st.subheader("📋 Prediction Results")

        st.dataframe(
            result_df.head(20),
            use_container_width=True
        )

        st.markdown("---")

        # =====================================================
        # SUMMARY
        # =====================================================

        total = len(result_df)

        normal = (result_df["Prediction"] == 0).sum()

        attack = (result_df["Prediction"] == 1).sum()

        attack_percent = round((attack / total) * 100, 2)

        st.subheader("📊 Prediction Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Records",
            f"{total:,}"
        )

        c2.metric(
            "Normal",
            f"{normal:,}"
        )

        c3.metric(
            "Attack",
            f"{attack:,}"
        )

        c4.metric(
            "Attack %",
            f"{attack_percent}%"
        )

        st.markdown("---")

        # =====================================================
        # DOWNLOAD BUTTON
        # =====================================================

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("---")

        # =====================================================
        # VISUALIZATIONS
        # =====================================================

        st.subheader("📈 Prediction Analytics")

        col1, col2 = st.columns(2)

                # =====================================
        # PIE CHART
        # =====================================

        with col1:

            pie_df = pd.DataFrame({

                "Traffic": ["Normal", "Attack"],

                "Count": [normal, attack]

            })

            fig = px.pie(

                pie_df,

                names="Traffic",

                values="Count",

                hole=0.45,

                title="Traffic Distribution"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.bar(
                pie_df,
                x="Traffic",
                y="Count",
                color="Traffic",
                text="Count",
                title="Prediction Count"
            )

            fig.update_layout(
            height=500
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown("---")

        # =====================================================
        # MODEL COMPARISON
        # =====================================================

        st.header("🏆 Model Performance Comparison")

        X_test = pd.read_csv("data/processed/X_test.csv")

        y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

        comparison_df = compare_models(
            X_test,
            y_test
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📊 Accuracy Comparison")

        fig = px.bar(
            comparison_df,
            x="Model",
            y="Accuracy",
            color="Model",
            text="Accuracy"
        )

        fig.update_layout(
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📈 Precision vs Recall vs F1")

        metrics = comparison_df.melt(
            id_vars="Model",
            value_vars=[
                "Precision",
                "Recall",
                "F1 Score"
            ],
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            metrics,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group"
        )

        fig.update_layout(
            height=550
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        best_model = comparison_df.loc[
            comparison_df["Accuracy"].idxmax()
        ]

        st.success(
            f"""
        🏆 Best Performing Model

        Model : {best_model['Model']}

        Accuracy : {best_model['Accuracy']:.4f}

        Precision : {best_model['Precision']:.4f}

        Recall : {best_model['Recall']:.4f}

        F1 Score : {best_model['F1 Score']:.4f}

        ROC AUC : {best_model['ROC AUC']:.4f}
        """
        )