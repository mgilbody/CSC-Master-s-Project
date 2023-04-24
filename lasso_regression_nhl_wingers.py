import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler

def lasso_regression(a):
    a.fillna(a.mean(), inplace=True)
    print(a.describe())

    X = a.drop(['Player', 'Tm', 'Pos', 'CAPHIT'], axis=1)
    y = a['CAPHIT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    lasso_model = Lasso(alpha= 0.1)
    lasso_model.fit(X_train, y_train)

    salary_prediction = lasso_model.predict(X_test)

    lasso_coef = list(zip(X, lasso_model.coef_))
    print(lasso_coef)

    difference = y_test - salary_prediction

    outcome = pd.DataFrame({'Salary': y_test, 'Predicted Salary': salary_prediction, 'Difference of Salaries': difference})
    print(outcome.head())

    alpha_score = LassoCV(cv= 5, random_state= 0)
    alpha_score.fit(X_train, y_train)

    print(alpha_score.alpha_)

    new_model = Lasso(alpha= alpha_score.alpha_)
    new_model.fit(X_train, y_train)
    new_coef = list(zip(X, new_model.coef_))
    print(new_coef)

    best_features = [x[0] for x in new_coef if x[1] != 0.0]
    print(best_features)

    new_salary_prediction = new_model.predict(X_test)

    difference = y_test - salary_prediction
    difference_2 = y_test - new_salary_prediction

    outcome = pd.DataFrame({'Salary': y_test, 'First Prediction': salary_prediction, 'First Difference': difference, 'New Prediction': new_salary_prediction, 'Second Difference': difference_2})
    print(outcome.head())

    print("First R^2 = " + str(sm.r2_score(y_test, salary_prediction)))
    print("Second R^2 = " + str(sm.r2_score(y_test, new_salary_prediction)))



player_stats = pd.read_csv('C:\\Users\\Seans Luceros Laptop\\Desktop\\nhl_stats_wingers.csv', encoding='latin1')
lasso_regression(player_stats)
