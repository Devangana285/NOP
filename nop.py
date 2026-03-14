import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Housing.csv")

print(df.head())



df = pd.get_dummies(df, drop_first=True)



X = df.drop("price", axis=1).values
y = df["price"].values



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def soft_threshold(beta, lam):

    return np.sign(beta) * np.maximum(np.abs(beta) - lam, 0)


def dynamic_soft_threshold(X, y, lr=0.01, lam=0.1, iterations=500):

    n, d = X.shape
    beta = np.zeros(d)

    loss_history = []

    for i in range(iterations):

        gradient = (1/n) * X.T.dot(X.dot(beta) - y)

        beta_temp = beta - lr * gradient

        dynamic_lambda = lam / (1 + 0.01*i)

        beta = soft_threshold(beta_temp, lr * dynamic_lambda)

        loss = np.mean((y - X.dot(beta))**2)

        loss_history.append(loss)

    return beta, loss_history

start = time.time()

beta, loss_history = dynamic_soft_threshold(X_train, y_train)

dynamic_time = time.time() - start

y_pred_dynamic = X_test.dot(beta)

dynamic_mse = mean_squared_error(y_test, y_pred_dynamic)

print("Dynamic Model MSE:", dynamic_mse)

start = time.time()

ridge = Ridge(alpha=1)

ridge.fit(X_train, y_train)

ridge_time = time.time() - start

ridge_pred = ridge.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_pred)

print("Ridge MSE:", ridge_mse)

start = time.time()

lasso = Lasso(alpha=0.1)

lasso.fit(X_train, y_train)

lasso_time = time.time() - start

lasso_pred = lasso.predict(X_test)

lasso_mse = mean_squared_error(y_test, lasso_pred)

print("LASSO MSE:", lasso_mse)

models = ["Ridge", "LASSO", "Dynamic"]

errors = [ridge_mse, lasso_mse, dynamic_mse]

plt.figure()

plt.bar(models, errors)

plt.title("Model Comparison (MSE)")

plt.ylabel("MSE")

plt.show()

plt.plot(loss_history)

plt.title("Optimization Convergence")

plt.xlabel("Iterations")

plt.ylabel("Loss")

plt.show()

plt.scatter(y_test, y_pred_dynamic)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted")

plt.show()