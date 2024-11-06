import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb


file_name = 'jamb_exam_results.csv'

df = pd.read_csv(file_name)
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.drop(columns=['student_id'])
df = df.fillna(0)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)

y_train = df_train['jamb_score']
y_val = df_val['jamb_score']
y_test = df_test['jamb_score']

del df_train['jamb_score']
del df_val['jamb_score']
del df_test['jamb_score']

dv = DictVectorizer(sparse=False)

train_dicts = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)

print(export_text(dt, feature_names=dv.get_feature_names_out()))

rf = RandomForestRegressor(n_estimators=10, random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(rmse)

for n in range(10,201,10):
    rf = RandomForestRegressor(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(n, round(rmse,3))

for m in [10, 15, 20, 25]:
    mean_rmse = 0
    for n in range(10,201,10):
        rf = RandomForestRegressor(n_estimators=n, max_depth=m, random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        mean_rmse += np.sqrt(mean_squared_error(y_val, y_pred))
    mean_rmse = mean_rmse / 20
    print(m, round(mean_rmse,3))    
    
rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1)
rf.fit(X_train, y_train)
features = dv.get_feature_names_out().tolist()
f_importance = pd.Series(rf.feature_importances_, index = features).sort_values(ascending=False)
print(f_importance)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

xgb_params = {
    'eta': 0.1, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

watchlist = [(dtrain, 'train'), (dval,'val')]

model = xgb.train(xgb_params, dtrain, evals=watchlist, num_boost_round=100)
y_pred = model.predict(dval)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(rmse)