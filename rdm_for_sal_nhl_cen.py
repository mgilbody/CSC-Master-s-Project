import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as sm
import seaborn as sns
from sklearn.tree import plot_tree



#read in data set using pandas
player_stats = pd.read_csv('C:\\Users\\Seans Luceros Laptop\\Desktop\\nhl_stats_centers.csv', encoding='latin1')

#fills in any missing values
player_stats.fillna(player_stats.mean(), inplace = True)

#X represents all numerical stats except for cap hit
#y reprsents cap hit since that's what is being predicted
X = player_stats.drop(['Player', 'Tm', 'Pos', 'CAPHIT'], axis= 1)
y = player_stats['CAPHIT']

#splitting the data, 80% of it will be training and 20% will be our test data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
predictor = RandomForestRegressor(n_estimators = 100, random_state = 0)
predictor.fit(X_train, y_train)
sal_predict = predictor.predict(X_val)

best_stats = pd.Series(predictor.feature_importances_, index= ['Age', 'GP', 'G', 'A', 'PTS', 'PLMI', 'PIM', 'PS',
                                'EV', 'STPP', 'STSH', 'GW', 'EVA', 'PPA', 'SHA', 'SH',
                                'SHP', 'TTOI', 'BLK', 'HIT', 'CF', 'CA', 'CFP', 'CFPR', 'FF', 'FA',
                                'FFP', 'FFPR', 'OISH', 'OISV', 'PDO', 'OZS', 'DZS', 'TK', 'GV', 'EPLMI',
                                'TSA', 'THRUP', 'FOL', 'FOW', 'FOP']).sort_values(ascending= False)

sns.barplot(x= best_stats, y= best_stats.index)
plt.title('Best NHL Stats (Centers)')
plt.show()

X_train = X_train.drop(['STSH', 'SHA'], axis= 1)
X_val = X_val.drop(['STSH', 'SHA'], axis= 1)

new_prediction = RandomForestRegressor(n_estimators= 100, random_state= 0)
new_prediction.fit(X_train, y_train)
improve_pred = new_prediction.predict(X_val)


diff_1 = sal_predict - y_val
diff_2 = improve_pred - y_val
new_outcome = pd.DataFrame({'Salary': y_val, 'First Prediction': sal_predict, 'Difference 1': diff_1, 'New Prediction': improve_pred, 'Difference 2': diff_2})
print("------------------------------------")
print("First Prediction vs New Prediction")
print(new_outcome.head())
print("------------------------------------")
print("First R^2 value: " + str((sm.r2_score(y_val, sal_predict))))
print("New R^2 value: " + str((sm.r2_score(y_val, improve_pred))))

plot_tree(predictor.estimators_[5])
plt.show()
plot_tree(new_prediction.estimators_[5])
plt.show()

errors = abs(sal_predict - y_val)
errors_2 = abs(improve_pred - y_val)

err_pct_1 = 100 * (errors/y_val)
err_pct_2 = 100 * (errors_2/y_val)

accuracy_1 = 100 - np.mean(err_pct_1)
accuracy_2 = 100 - np.mean(err_pct_2)

print("Accuracy score 1: ", round(accuracy_1, 2), "%")
print("Accuracy score 2: ", round(accuracy_2, 2), "%")

mse = sm.mean_absolute_error(y_val, sal_predict)
mse2 = sm.mean_absolute_error(y_val, improve_pred)

print("Mean absolute error 1: ", mse)
print("Mean absolute error 2: ", mse2)
