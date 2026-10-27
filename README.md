# EV Car Price Prediction Using Ridge Regression

## 📌 Project Overview

This project focuses on predicting the prices of Electric Vehicles (EVs) in India using Machine Learning.

The project implements **Ridge Regression** with a complete preprocessing pipeline to handle both categorical and numerical features.

- **Categorical Features:** Brand and Model are processed using One-Hot Encoding.
- **Numerical Features:** Range, Power, and Battery are standardized using StandardScaler.
- **Model:** Ridge Regression with different regularization values (alpha).
- **Evaluation:** MAE, RMSE, and R² Score.

## 🎯 Objectives

- Analyze an Electric Vehicle dataset.
- Prepare categorical and numerical features for Machine Learning.
- Predict EV prices based on vehicle specifications.
- Implement Ridge Regression.
- Experiment with different Ridge regularization values.
- Evaluate model performance using regression metrics.
- Build a reusable Machine Learning preprocessing pipeline.

## 📊 Dataset

The project uses an EV dataset containing information about Electric Vehicles in India.

### Dataset Features

| Feature | Description |
|---|---|
| Brand | Brand or manufacturer of the EV |
| Model | Model name of the EV |
| Price | Price of the EV (Target Variable) |
| Range | Driving range of the vehicle |
| Power | Power of the vehicle |
| Battery | Battery capacity |

### Target Variable

- `Price`

### Input Features

- `Brand`
- `Model`
- `Range`
- `Power`
- `Battery`

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 🤖 Machine Learning Algorithm

### Ridge Regression

Ridge Regression is a linear regression algorithm that uses **L2 regularization** to reduce model complexity and help prevent overfitting.

The project tests multiple values of the regularization parameter:

```python
alphas = [0.01, 0.1, 1, 10, 100]
```

This allows the model's performance to be compared across different regularization strengths.

## 🔄 Machine Learning Workflow

```text
                 EV Dataset
                      |
                      v
                Load Dataset
                      |
                      v
               Data Exploration
                      |
                      v
               Feature Selection
                      |
          +-----------+-----------+
          |                       |
          v                       v
    Categorical              Numerical
      Features                Features
          |                       |
          v                       v
    One-Hot Encoding       Standard Scaling
          |                       |
          +-----------+-----------+
                      |
                      v
              Preprocessing
                  Pipeline
                      |
                      v
               Train-Test Split
                      |
                      v
              Ridge Regression
                      |
                      v
           Test Different Alpha Values
                      |
                      v
                 Predictions
                      |
                      v
               Model Evaluation
                      |
                      v
                 MAE / RMSE / R²
```

## 🧹 Data Preprocessing

The dataset is divided into categorical and numerical features.

### 1. Categorical Features

```python
categorical = ["Brand", "Model"]
```

These features are transformed using One-Hot Encoding:

```python
OneHotEncoder(handle_unknown="ignore")
```

This converts categorical values into numerical representations suitable for Machine Learning.

### 2. Numerical Features

```python
numerical = ["Range", "Power", "Battery"]
```

These features are standardized using:

```python
StandardScaler()
```

Standardization helps numerical features use a consistent scale.

### 3. Column Transformer

Both preprocessing operations are combined using `ColumnTransformer`:

```python
preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical),

    ("numerical",
     StandardScaler(),
     numerical)
])
```

## ✂️ Train-Test Split

The dataset is divided into training and testing data using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The dataset is split as follows:

- **80% Training Data**
- **20% Testing Data**

The training data is used to train the model, while the testing data is used to evaluate its performance.

## 🔗 Machine Learning Pipeline

The project uses a Scikit-learn `Pipeline` to combine preprocessing and Ridge Regression.

```python
model = Pipeline([
    ("preprocessing", preprocessor),
    ("ridge", Ridge(alpha=alpha))
])
```

This pipeline automatically applies preprocessing before model training and prediction, helping maintain a consistent workflow.

## 📈 Model Evaluation

The model is evaluated using three regression metrics.

### 1. Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted prices.

```python
mean_absolute_error(y_test, test_pred)
```

### 2. Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

```python
np.sqrt(mean_squared_error(y_test, test_pred))
```

### 3. R² Score

R² Score measures how well the model explains the variation in the target variable.

```python
r2_score(y_test, test_pred)
```

Lower MAE and RMSE values generally indicate better prediction accuracy. An R² value closer to 1 generally indicates a better fit.

## 🧪 Ridge Regression Parameters

The project evaluates the following alpha values:

| Alpha |
|---:|
| 0.01 |
| 0.1 |
| 1 |
| 10 |
| 100 |

The purpose is to compare how different regularization strengths affect the Ridge Regression model.

## 📋 Results

The notebook is designed to generate a results table containing the following metrics:

| Alpha | Train MAE | Train RMSE | Train R² | Test MAE | Test RMSE | Test R² |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | — | — | — | — | — | — |
| 0.1 | — | — | — | — | — | — |
| 1 | — | — | — | — | — | — |
| 10 | — | — | — | — | — | — |
| 100 | — | — | — | — | — | — |

**Note:** Numerical results are not included because the model evaluation metrics need to be calculated in the notebook. The values should be filled in after running the model.

## 📁 Project Structure

```text
ML-Task-9/
│
├── ev_car_India_dataset.csv
├── EV_Car_Price_Prediction.ipynb
└── README.md
```

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Rabinson-20/ML-Task-9.git
```

### 2. Navigate to the Project Directory

```bash
cd ML-Task-9
```

### 3. Install the Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the Notebook

Open the following file:

`EV_Car_Price_Prediction.ipynb`

Run the cells sequentially to perform:

1. Dataset loading
2. Data exploration
3. Feature selection
4. Data preprocessing
5. Train-test splitting
6. Ridge Regression model training
7. Prediction
8. Model evaluation
9. Results comparison

## 💡 Key Concepts Demonstrated

This project demonstrates practical implementation of:

- Data loading using Pandas
- Dataset exploration
- Missing-value checking
- Feature selection
- Categorical feature encoding
- Numerical feature scaling
- Train-test splitting
- Scikit-learn ColumnTransformer
- Scikit-learn Pipeline
- Ridge Regression
- Hyperparameter experimentation
- Regression evaluation metrics
- MAE, RMSE, and R² Score

## 🚀 Future Improvements

The project can be extended by:

- Increasing the size of the EV dataset.
- Performing detailed Exploratory Data Analysis (EDA).
- Adding more vehicle features.
- Using GridSearchCV for automatic hyperparameter tuning.
- Applying cross-validation.
- Comparing Ridge Regression with other algorithms, such as:
  - Linear Regression
  - Lasso Regression
  - Random Forest Regression
  - Gradient Boosting
- Creating visualizations for EV prices and specifications.
- Developing an interactive EV price prediction application using Streamlit.
- Deploying the Machine Learning model as a web application.

## 🏁 Conclusion

This project demonstrates how Machine Learning can be used to predict Electric Vehicle prices based on specifications such as brand, model, driving range, power, and battery capacity.

By combining One-Hot Encoding, StandardScaler, ColumnTransformer, Pipeline, and Ridge Regression, the project provides a structured approach to building a regression model for EV price prediction.
