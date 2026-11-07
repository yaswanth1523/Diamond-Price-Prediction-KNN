
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------------
# 1. Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="wide"
)

# -----------------------------------
# 2. Load trained model
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "knn_diamond_price_model.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# -----------------------------------
# 3. Input validation function
# -----------------------------------


def validate_diamond_input(carat, depth, table, x, y, z):

    errors = []

    if carat <= 0:
        errors.append("Carat must be greater than zero.")

    if depth <= 0:
        errors.append("Depth must be greater than zero.")

    if table <= 0:
        errors.append("Table must be greater than zero.")

    if min(x, y, z) <= 0:
        errors.append("All diamond dimensions must be positive.")

    if max(x, y, z) > 15:
        errors.append("Diamond dimensions must not exceed 15 mm.")

    return errors


# -----------------------------------
# 4. Application heading
# -----------------------------------

st.title("💎 Diamond Price Prediction")

st.write(
    "Predict the estimated price of a diamond "
    "using its physical and quality characteristics."
)

st.divider()

# -----------------------------------
# 5. User input form
# -----------------------------------

st.subheader("Enter Diamond Characteristics")

with st.form("diamond_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        carat = st.number_input(
            "Carat",
            min_value=0.20,
            max_value=5.01,
            value=1.00,
            step=0.01
        )

        cut = st.selectbox(
            "Cut",
            ["Fair", "Good", "Very Good", "Premium", "Ideal"]
        )

        color = st.selectbox(
            "Color",
            ["J", "I", "H", "G", "F", "E", "D"]
        )

    with col2:

        clarity = st.selectbox(
            "Clarity",
            ["I1", "SI2", "SI1", "VS2",
             "VS1", "VVS2", "VVS1", "IF"]
        )

        depth = st.number_input(
            "Depth (%)",
            min_value=43.0,
            max_value=79.0,
            value=61.5,
            step=0.1
        )

        table = st.number_input(
            "Table (%)",
            min_value=43.0,
            max_value=95.0,
            value=57.0,
            step=0.1
        )

    with col3:

        x = st.number_input(
            "Length (x) in mm",
            min_value=0.01,
            max_value=15.0,
            value=5.5,
            step=0.01
        )

        y = st.number_input(
            "Width (y) in mm",
            min_value=0.01,
            max_value=15.0,
            value=5.5,
            step=0.01
        )

        z = st.number_input(
            "Depth (z) in mm",
            min_value=0.01,
            max_value=15.0,
            value=3.4,
            step=0.01
        )

    submitted = st.form_submit_button(
        "Predict Diamond Price",
        use_container_width=True
    )


# -----------------------------------
# 6. Prediction
# -----------------------------------

if submitted:

    # Validate inputs
    errors = validate_diamond_input(
        carat, depth, table, x, y, z
    )

    if errors:

        for error in errors:
            st.error(error)

    else:

        # Calculate engineered feature
        volume = x * y * z

        # Prepare input DataFrame
        input_data = pd.DataFrame([{
            "carat": carat,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
            "volume": volume
        }])

        # Generate prediction
        predicted_price = model.predict(input_data)[0]

        st.divider()

        st.success("Prediction generated successfully!")

        st.metric(
            label="Estimated Diamond Price",
            value=f"${predicted_price:,.2f}"
        )

        st.caption(
            "This is a machine learning estimate, "
            "not a guaranteed market price."
        )

        with st.expander("View submitted diamond details"):
            st.dataframe(
                input_data,
                use_container_width=True
            )


# -----------------------------------
# 7. About the project
# -----------------------------------

st.divider()

st.subheader("About This Project")

st.write("""
This application uses a Machine Learning model based on
K-Nearest Neighbors (KNN) to estimate diamond prices.

**Machine Learning Algorithm:** KNN Regression

**Model Configuration:**
- Neighbors: 9
- Distance Weighting: Distance
- Distance Metric: Manhattan

**Preprocessing:**
- StandardScaler
- OrdinalEncoder

**Evaluation Results:**
- MAE: $315.82
- RMSE: $617.62
- R²: 0.9764

Developed using Python, Pandas, Scikit-learn and Streamlit.
""")

st.caption(
    "Educational machine learning project. "
    "Predictions are estimates, not guaranteed market prices."
)
