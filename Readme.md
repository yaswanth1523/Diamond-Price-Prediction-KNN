# 💎 Diamond Price Prediction Using K-Nearest Neighbors (KNN)

An end-to-end Machine Learning project that predicts diamond prices based on physical dimensions and quality characteristics using a tuned K-Nearest Neighbors (KNN) regression model.

The project includes data inspection, exploratory data analysis, data cleaning, feature engineering, preprocessing, model development, hyperparameter tuning, evaluation, and an interactive Streamlit application.

## 🎯 Project Objective

To develop a Machine Learning model that estimates the price of a diamond using characteristics such as carat, cut, color, clarity, depth, table, and physical dimensions.

## 📊 Dataset

- **Dataset:** Diamonds
- **Records:** 53,940
- **Features:** 9 original input features
- **Target:** Price (USD)
- **Source:** ggplot2 Diamonds Dataset

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## ⚙️ Project Workflow

1. Data loading and inspection
2. Exploratory Data Analysis (EDA)
3. Missing-value and duplicate analysis
4. Investigation of invalid physical measurements
5. Train-test splitting with duplicate groups kept together
6. Data cleaning
7. Feature engineering
8. Numerical scaling and categorical encoding
9. KNN regression model development
10. Hyperparameter tuning using GridSearchCV
11. Cross-validation and final test evaluation
12. Prediction and residual analysis
13. Model serialization using Joblib
14. Streamlit application development

## 🧠 Model Development

**Algorithm:** K-Nearest Neighbors Regressor

**Best Hyperparameters:**

| Parameter | Value |
|---|---|
| Number of neighbors | 9 |
| Weights | Distance |
| Distance metric | Manhattan |
| Minkowski parameter (p) | 1 |

**Preprocessing:**
- StandardScaler for numerical features
- OrdinalEncoder for categorical features
- Volume feature engineering: x × y × z

## 📈 Model Performance

Final held-out test results:

| Metric | Score |
|---|---:|
| MAE | $315.82 |
| RMSE | $617.62 |
| R² | 0.9764 |

The model achieved an R² score of 0.9764 on the held-out test dataset.

## 💻 Streamlit Application

The application allows users to enter diamond characteristics and receive an estimated price.

Features:
- Interactive diamond input form
- Automatic volume calculation
- KNN-based price prediction
- Display of submitted diamond details
- Model information and evaluation metrics

## 📁 Project Structure

```text
Diamond_Price_Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── diamonds.csv
│   └── cleaned/
│
├── models/
│   └── knn_diamond_price_model.joblib
│
├── notebooks/
│   └── 01_data_inspection_EDA.ipynb
│
├── reports/
├── screenshots/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Diamond_Price_Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the application

```bash
streamlit run app/app.py
```

## ⚠️ Disclaimer

The predicted price is a Machine Learning estimate based on the dataset and model used in this project. It is not a guaranteed market price or professional diamond valuation.

## 🌐 Live Demo

Try the deployed application:

[Diamond Price Prediction App](https://diamond-price-prediction-knn.streamlit.app/)

## 📂 GitHub Repository

[View Source Code](https://github.com/yaswanth1523/Diamond-Price-Prediction-KNN)

## 👨‍💻 Author

**Yaswanth Kumar Reddy M**

Machine Learning | Data Science | Python | Data Analytics
