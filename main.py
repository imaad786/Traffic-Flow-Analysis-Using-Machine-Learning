# import the required modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
(X, Y) = (train_df.drop(['traffic_volume'], axis = 1).values, train_df['traffic_volume'].values)
# Scale the values
scaler = StandardScaler()
X = scaler.fit_transform(X)

(X_train, X_val, Y_train, Y_val) = train_test_split(X, Y)
print("X_train shape:" + str(X_train.shape))
print("Y_train shape:" + str(Y_train.shape))
print("X_val shape:" + str(X_val.shape))
print("Y_val shape:" + str(Y_val.shape))
# DataFrame to store the RMSE scores of various algorithms
results = pd.DataFrame(columns = ['RMSE'])
# helper function to evaluate a model
def evaluate_model(regressor, name):
    # train and test scores
    train_score = round(regressor.score(X_train, Y_train), 2)
    val_score = round(regressor.score(X_val, Y_val), 2)
    # predicted output
    Y_pred = regressor.predict(X_val)



    print(name + ' Train score: ', train_score)
    print(name + 'Test score: ', val_score)
    print('Root Mean Squared error: ', sqrt(mean_squared_error(Y_val, Y_pred)))
    print('Coefficient of determination: ', r2_score(Y_val, Y_pred))
    

    # add the current RMSE to the scores list
    results.loc[name] = sqrt(mean_squared_error(Y_val, Y_pred))
    
    # plot predicted vs true values
    x_points=np.linspace(0,8e3)
    plt.figure(figsize=(12,5))
    plt.plot(x_points, x_points, color='r')
    plt.scatter(Y_val, Y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True Values Vs Predicted Values');
lireg = LinearRegression()
lireg.fit(X_train, Y_train)
# evaluate the Regressor
evaluate_model(lireg, 'Linear Regression')

dtreg = DecisionTreeRegressor(max_depth = 12)
dtreg.fit(X_train, Y_train)
# evaluate the Regressor
evaluate_model(dtreg, 'Decision Tree')
# n_estimators - The number of trees in the forest.
# min_samples_split - The minimum number of samples required to split an internal node
rfreg = RandomForestRegressor(n_estimators = 60, max_depth = 13, min_samples_split = 5)
rfreg.fit(X_train, Y_train)
# evaluate the Regressor
evaluate_model(rfreg, 'Random Forest')
# n_estimators - The number of boosting stages to perform.
# max_depth - maximum depth of the individual regression estimators.
gbreg = GradientBoostingRegressor(n_estimators=497, max_depth=10)
gbreg.fit(X_train, Y_train)
# evaluate the Regressor
evaluate_model(gbreg, 'Gradient Boosting')
# n_estimators - The number of trees in the forest.
# learning_rate - Learning rate shrinks the contribution of each classifier by learning_rate.
adareg = AdaBoostRegressor(base_estimator=dtreg, n_estimators=60, learning_rate=0.005)
adareg.fit(X_train, Y_train)
# evaluate the Regressor
evaluate_model(adareg, 'Ada Boost')
##results
plt.plot(gbreg.feature_importances_)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline

def nn_model ():
    model = Sequential()
    model.add(Dense(128, input_dim=10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))




    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
estimator = KerasRegressor(build_fn=nn_model, epochs=10, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
estimator.fit(X_train, Y_train)
# predicted output
Y_pred_nn = estimator.predict(X_val)

print('Root Mean Squared error: ', sqrt(mean_squared_error(Y_val, Y_pred_nn)))
print('Coefficient of determination: ', r2_score(Y_val, Y_pred_nn))
train_df['holiday'].value_counts()
z = lambda x: False if x == 'None' else True
train_df['holiday'] = train_df['holiday'].apply(z)
fig, (axis1,axis2) = plt.subplots(1, 2, figsize = (20,6))
sns.countplot(x = 'holiday', data = train_df, ax = axis1)
sns.barplot(x = 'holiday', y = 'traffic_volume', data = train_df, ax = axis2);

(train_df['temp'] == 0).sum()
train_df = train_df[train_df['temp'] != 0]
sns.scatterplot(x = 'temp', y = 'traffic_volume', data = train_df);

(train_df['rain_1h'] > 100).sum()
train_df = train_df[train_df.rain_1h < 100]
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.scatterplot(x = 'rain_1h', y = 'traffic_volume', data = train_df,);

sns.scatterplot(x = 'snow_1h', y = 'traffic_volume', data = train_df);
sns.scatterplot(x = 'clouds_all', y = 'traffic_volume', data = train_df);
### Year vs Traffic Volume
train_df['year'] = train_df['date_time'].dt.year
fig, (axis1,axis2) = plt.subplots(1, 2, figsize = (20,6))
sns.countplot(x = 'year', data = train_df, ax = axis1, palette="Set2")
sns.lineplot(x = 'year', y = 'traffic_volume', data = train_df, ax = axis2);
### Month vs Traffic Volume
train_df['month'] = train_df['date_time'].dt.month
fig, (axis1,axis2) = plt.subplots(2, 1, figsize = (20,12))
sns.countplot(x = 'month', data = train_df, ax = axis1, palette="Set3")
sns.lineplot(x = 'month', y = 'traffic_volume', data = train_df, ax = axis2,)
### Day vs Traffic Volume

train_df['day'] = train_df['date_time'].dt.day_name()
fig, (axis1,axis2) = plt.subplots(1, 2, figsize = (20,6))
sns.countplot(x = 'day', data = train_df, ax = axis1)
sns.lineplot(x = 'day', y = 'traffic_volume', data = train_df, ax = axis2);
##Short Weather Description vs Traffic Volume

fig, (axis1,axis2) = plt.subplots(2, 1, figsize = (16,12))
sns.countplot(x = 'weather_main', data = train_df, ax = axis1)
sns.lineplot(x = 'weather_main', y = 'traffic_volume', data = train_df, ax = axis2);

train_df['weather_description'].value_counts()
plt.figure(figsize = (20,6))
sns.lineplot(x = 'weather_description', y = 'traffic_volume', data = train_df);
##correlation
plt.figure(figsize=(8, 5))
plt.title('Correlation between features')
sns.heatmap(train_df.corr(), annot = True);
