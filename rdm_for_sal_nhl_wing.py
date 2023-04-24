import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as sm
import seaborn as sns
from sklearn.tree import plot_tree

#read in data set using pandas
player_stats = pd.read_csv('C:\\Users\\Seans Luceros Laptop\\Desktop\\nhl_stats_wingers.csv', encoding='latin1')

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
                                'TSA', 'THRUP']).sort_values(ascending= False)

#creates the plot that ranks the features
sns.barplot(x= best_stats, y= best_stats.index)
plt.title('Best NHL Stats')
plt.show()

#creating new random forest from dropping the least important features
X_train = X_train.drop(['SHA'], axis= 1)
X_val = X_val.drop(['SHA'], axis= 1)

new_prediction = RandomForestRegressor(n_estimators= 100, random_state= 0)
new_prediction.fit(X_train, y_train)
improve_pred = new_prediction.predict(X_val)

#shows us the actual salary, first predicted salary, difference between first and actual, second predicted salary, difference between second and actual
diff_1 = sal_predict - y_val
diff_2 = improve_pred - y_val
new_outcome = pd.DataFrame({'Salary': y_val, 'First Prediction': sal_predict, 'Difference 1': diff_1, 'New Prediction': improve_pred, 'Difference 2': diff_2})
print("------------------------------------")
print("First Prediction vs New Prediction")
print(new_outcome.head())
print("First R^2 value: " + str((sm.r2_score(y_val, sal_predict))))
print("New R^2 value: " + str((sm.r2_score(y_val, improve_pred))))

#prints out an example of a decision tree from the random forest
plot_tree(predictor.estimators_[5])
plt.show()
plot_tree(new_prediction.estimators_[5])
plt.show()
