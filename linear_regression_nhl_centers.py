import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler

def multiple_linear_regression(a):
    a.fillna(a.mean(), inplace=True)
    print(a.describe())

    X = a.drop(['Player', 'Tm', 'Pos', 'CAPHIT'], axis=1)
    y = a['CAPHIT']
    np.column_stack((X, y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    print("The intercept is: " + str(lr_model.intercept_))
    print("The coefficients are: ")
    print(list(zip(X, lr_model.coef_)))

    sal_predict = lr_model.predict(X_test)
    difference = y_test - sal_predict

    outcome = pd.DataFrame({'Salary': y_test, 'Predicted Salary': sal_predict, 'Difference of Salaries': difference})

    print(outcome)

    print(a.corr())

    X_train = X_train.drop(['PLMI', 'PIM', 'STSH', 'SHA', 'HIT', 'PDO', 'THRUP'], axis=1)
    X_test = X_test.drop(['PLMI', 'PIM', 'STSH', 'SHA', 'HIT', 'PDO', 'THRUP'], axis=1)

    new_predict = LinearRegression()
    new_predict.fit(X_train, y_train)

    print("The intercept is: " + str(lr_model.intercept_))
    print("The coefficients are: ")
    print(list(zip(X, lr_model.coef_)))

    second_prediction = new_predict.predict(X_test)

    difference_2 = y_test - second_prediction
    outcome_2 = pd.DataFrame({'Salary': y_test, 'First Prediction': sal_predict, 'Difference 1': difference, 'Second Prediction': second_prediction, 'Difference 2': difference_2})
    print(outcome_2.head())
    print("First R^2 = " + str(sm.r2_score(y_test, sal_predict)))
    print("New R^2 = " + str(sm.r2_score(y_test, second_prediction)))
    
    errors = abs(sal_predict - y_test)
    errors_2 = abs(second_prediction - y_test)

    err_pct_1 = 100 * (errors / y_test)
    err_pct_2 = 100 * (errors_2 / y_test)

    accuracy_1 = 100 - np.mean(err_pct_1)
    accuracy_2 = 100 - np.mean(err_pct_2)

    print("Accuracy score 1: ", round(accuracy_1, 2), "%")
    print("Accuracy score 2: ", round(accuracy_2, 2), "%")

    mse = sm.mean_absolute_error(y_test, sal_predict)
    mse2 = sm.mean_absolute_error(y_test, second_prediction)

    print("Mean absolute error 1: ", mse)
    print("Mean absolute error 2: ", mse2)


player_stats = pd.read_csv('C:\\Users\\Seans Luceros Laptop\\Desktop\\nhl_stats_centers.csv', encoding='latin1')
multiple_linear_regression(player_stats)
