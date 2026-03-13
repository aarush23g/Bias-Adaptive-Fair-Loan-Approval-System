import os
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Bias-Adaptive Fair Loan Approval Dashboard",
    layout="wide"
)

# --------------------------------------------------
# Base Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_TABLE_DIR = os.path.join(BASE_DIR, "results", "tables")
RESULTS_FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
RESULTS_STATS_DIR = os.path.join(BASE_DIR, "results", "statistics")


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def safe_read_csv(path):
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)

        # If file exists but has no rows
        if df.empty:
            return pd.DataFrame()

        return df

    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error reading file {path}: {e}")
        return None


def show_image(path, caption=None):
    if os.path.exists(path):
        image = Image.open(path)
        st.image(image, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {path}")


def show_table(title, path):
    st.subheader(title)
    df = safe_read_csv(path)

    if df is None:
        st.warning(f"Table not found: {path}")
    elif df.empty:
        st.info("No data available for this table.")
    else:
        st.dataframe(df, use_container_width=True)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Final Comparison",
        "Dataset-wise Results",
        "Tradeoff Analysis",
        "Controller Dynamics",
        "Statistical Stability",
        "Intersectional Fairness",
        "Explainability"
    ]
)

# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Overview":
    st.title("Bias-Adaptive Fair Loan Approval System")

    st.markdown("""
    This dashboard presents the results of the **Bias-Adaptive Fair Loan Approval System** project.

    ### Project Goal
    Build a fairness-aware machine learning framework for loan approval that:
    - predicts loan decisions accurately
    - measures fairness across sensitive groups
    - reduces bias using an **adaptive fairness controller**

    ### Datasets Used
    - **German Credit**
    - **LendingClub**
    - **Adult Income**

    ### Methods Compared
    - Logistic Regression
    - Random Forest
    - Fair Logistic Regression
    - Adaptive Controller (LightGBM)
    - Fairlearn Exponentiated Gradient
    - Fairlearn GridSearch
    - LightGBM Threshold Optimization

    ### Fairness Metrics
    - Demographic Parity (DP)
    - Equal Opportunity (EOP)
    - Equalized Odds (EOD)

    ### Main Contribution
    Instead of applying a fixed fairness penalty, this project introduces an
    **Adaptive Fairness Controller** that updates fairness strength dynamically
    during training based on observed fairness violations.
    """)

# --------------------------------------------------
# Final Comparison
# --------------------------------------------------
elif page == "Final Comparison":
    st.title("Final Method Comparison")

    path = os.path.join(RESULTS_TABLE_DIR, "final_method_comparison.csv")
    df = safe_read_csv(path)

    if df is not None:
        st.subheader("All Results")
        st.dataframe(df, use_container_width=True)

        st.subheader("Filter by Dataset")
        datasets = ["All"] + sorted(df["dataset"].dropna().unique().tolist())
        selected_dataset = st.selectbox("Dataset", datasets)

        if selected_dataset != "All":
            filtered_df = df[df["dataset"] == selected_dataset]
        else:
            filtered_df = df

        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning(f"File not found: {path}")

# --------------------------------------------------
# Dataset-wise Results
# --------------------------------------------------
elif page == "Dataset-wise Results":
    st.title("Dataset-wise Results")

    dataset = st.selectbox("Choose Dataset", ["German", "LendingClub", "Adult"])

    dataset_key_map = {
        "German": "german",
        "LendingClub": "lending_club",
        "Adult": "adult"
    }

    key = dataset_key_map[dataset]

    show_table(
        "Baseline Results",
        os.path.join(RESULTS_TABLE_DIR, f"{key}_baseline_metrics.csv")
    )
    show_table(
        "Static Fairness Results",
        os.path.join(RESULTS_TABLE_DIR, f"{key}_static_metrics.csv")
    )
    show_table(
        "Adaptive Controller Results",
        os.path.join(RESULTS_TABLE_DIR, f"{key}_adaptive_metrics.csv")
    )
    show_table(
        "Fairlearn Results",
        os.path.join(RESULTS_TABLE_DIR, f"{key}_fairlearn_metrics.csv")
    )
    show_table(
        "Threshold Optimization Results",
        os.path.join(RESULTS_TABLE_DIR, f"{key}_lightgbm_threshold_final.csv")
    )

# --------------------------------------------------
# Tradeoff Analysis
# --------------------------------------------------
elif page == "Tradeoff Analysis":
    st.title("Fairness–Accuracy Tradeoff")

    dataset = st.selectbox("Choose Dataset", ["German", "LendingClub", "Adult"])

    img_map = {
        "German": "fairness_accuracy_tradeoff_german.png",
        "LendingClub": "fairness_accuracy_tradeoff_lendingclub.png",
        "Adult": "fairness_accuracy_tradeoff_adult.png"
    }

    show_image(
        os.path.join(RESULTS_FIG_DIR, img_map[dataset]),
        caption=f"Fairness–Accuracy Tradeoff ({dataset})"
    )

# --------------------------------------------------
# Controller Dynamics
# --------------------------------------------------
elif page == "Controller Dynamics":
    st.title("Controller Dynamics")

    metric = st.selectbox(
        "Choose Fairness Metric",
        ["dp", "eop", "eod"]
    )

    show_image(
        os.path.join(RESULTS_FIG_DIR, f"controller_dynamics_{metric}.png"),
        caption=f"Controller Dynamics ({metric.upper()})"
    )

# --------------------------------------------------
# Statistical Stability
# --------------------------------------------------
elif page == "Statistical Stability":
    st.title("Statistical Stability Analysis")

    path = os.path.join(RESULTS_STATS_DIR, "statistical_summary.csv")
    df = safe_read_csv(path)

    if df is not None:
        st.dataframe(df, use_container_width=True)
        st.markdown("""
        This table summarizes multi-seed experiments and reports:
        - mean accuracy
        - standard deviation
        - mean ROC-AUC
        - fairness stability
        """)
    else:
        st.warning(f"File not found: {path}")

# --------------------------------------------------
# Intersectional Fairness
# --------------------------------------------------
elif page == "Intersectional Fairness":
    st.title("Intersectional Fairness")

    show_table(
        "Intersectional Fairness Results",
        os.path.join(RESULTS_TABLE_DIR, "intersectional_fairness_results.csv")
    )

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            os.path.join(RESULTS_FIG_DIR, "german_fairness_bar.png"),
            caption="German Fairness Bar Plot"
        )

    with col2:
        show_image(
            os.path.join(RESULTS_FIG_DIR, "lending_club_fairness_bar.png"),
            caption="LendingClub Fairness Bar Plot"
        )

# --------------------------------------------------
# Explainability
# --------------------------------------------------
elif page == "Explainability":
    st.title("Explainability")

    dataset = st.selectbox("Choose Dataset", ["German", "LendingClub", "Adult"])

    shap_global_map = {
        "German": "shap_global_german.png",
        "LendingClub": "shap_global_lending_club.png",
        "Adult": "shap_global_adult.png"
    }

    shap_group_map = {
        "German": "shap_group_difference_german.png",
        "LendingClub": "shap_group_difference_lending_club.png",
        "Adult": "shap_group_difference_adult.png"
    }

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            os.path.join(RESULTS_FIG_DIR, shap_global_map[dataset]),
            caption=f"Global SHAP ({dataset})"
        )

    with col2:
        show_image(
            os.path.join(RESULTS_FIG_DIR, shap_group_map[dataset]),
            caption=f"Group SHAP Difference ({dataset})"
        )

    dataset_key_map = {
        "German": "german",
        "LendingClub": "lending_club",
        "Adult": "adult"
    }

    key = dataset_key_map[dataset]

    show_table(
        "Counterfactual Examples",
        os.path.join(RESULTS_TABLE_DIR, f"counterfactual_examples_{key}.csv")
    )