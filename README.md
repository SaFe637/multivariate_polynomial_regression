# Multivariate Polynomial Regression

A simple implementation of **Multivariate Polynomial Regression** using Python, NumPy, Pandas, Matplotlib, and Scikit-learn.

This project demonstrates how polynomial regression can be applied to a dataset with multiple input features. The model uses polynomial features of degree 2 and calculates the regression weights using the **Normal Equation**.

## Project Overview

In this project, a polynomial regression model is trained on an automotive dataset using multiple features.

The main steps of the project are:

1. Load the training and testing datasets.
2. Separate input features (`X`) and target values (`y`).
3. Explore the relationship between each feature and the target.
4. Generate polynomial features of degree 2.
5. Calculate the model weights using the Normal Equation.
6. Make predictions on the training and test sets.
7. Evaluate the model using MAE and R².
8. Visualize the predicted values compared with the actual values.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Dataset

The project uses two CSV files:

* `auto-train-multi.csv` — Training dataset
* `auto-test-multi.csv` — Testing dataset

The input features are separated from the final column, which is used as the target variable.

```python
x_train = train_set[:, :-1]
y_train = train_set[:, -1:]
```

The same process is applied to the test dataset.

## Polynomial Features

The model uses polynomial features with **degree 2**:

```python
poly = PolynomialFeatures(degree=2)
```

The training and testing data are transformed using:

```python
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
```

This allows the model to capture nonlinear relationships between the input features and the target variable.

## Model

The regression prediction is calculated using matrix multiplication:

```python
def linear_regression(x, w):
    y_hat = x @ w
    return y_hat
```

The weights are calculated using the **Normal Equation**:

```python
w = np.linalg.inv(x_train_poly.T @ x_train_poly) @ x_train_poly.T @ y_train
```

The model can then be used to predict both training and test data.

## Evaluation Metrics

Three evaluation functions are implemented in the notebook:

### Mean Squared Error (MSE)

```python
def mse(y, y_hat):
    loss = np.mean((y - y_hat)**2)
    return loss
```

### Mean Absolute Error (MAE)

```python
def mae(y, y_hat):
    loss = np.mean(np.abs(y - y_hat))
    return loss
```

### R² Score

```python
def r2(y, y_hat):
    return 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)
```

For the final test evaluation, **MAE** and **R²** are calculated.

## Visualization

The project also visualizes the actual training values against the model's predictions.

The plot includes an ideal fit line where:

**Actual Value = Predicted Value**

Points closer to this line indicate predictions that are closer to the actual values.

## Project Structure

```text
Polynomial-Regression/
│
├── polynomial_regression_multivariate.ipynb
├── auto-train-multi.csv
├── auto-test-multi.csv
└── README.md
```

## How to Run

1. Clone or download this repository.
2. Make sure the required CSV files are in the same directory as the notebook.
3. Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

4. Open the notebook:

```bash
jupyter notebook polynomial_regression_multivariate.ipynb
```

5. Run the cells from top to bottom.

## Learning Goals

This project was created to practice:

* Multivariate regression
* Polynomial feature transformation
* Matrix operations with NumPy
* The Normal Equation
* Model prediction
* Regression evaluation metrics
* Data visualization
* Working with training and testing datasets


