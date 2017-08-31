# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:04:18 2017

@author: Pedrors
"""



import pandas
import xgboost as xgb
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans


path = r'C:\Users\Pedrors\Desktop\Programação\kaggle\New York City Taxi Trip Duration'

train = pandas.read_csv(path+'\\train.csv')

# removing the outliers

mean = numpy.mean(train['trip_duration'])
std_dev = numpy.std(train['trip_duration'])
train = train[train['trip_duration'] <= mean + 2*std_dev]
train = train[train['trip_duration'] >= mean - 2*std_dev]

# removing points outside NY

train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]

train.head()

# this is from https://www.kaggle.com/karelrv/nyct-from-a-to-z-with-xgboost-tutorial

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(numpy.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = numpy.sin(lat * 0.5) ** 2 + numpy.cos(lat1) * numpy.cos(lat2) * numpy.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * numpy.arcsin(numpy.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = numpy.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(numpy.radians, (lat1, lng1, lat2, lng2))
    y = numpy.sin(lng_delta_rad) * numpy.cos(lat2)
    x = numpy.cos(lat1) * numpy.sin(lat2) - numpy.sin(lat1) * numpy.cos(lat2) * numpy.cos(lng_delta_rad)
    return numpy.degrees(numpy.arctan2(y, x))


# function to engineering some features both on train and test set

def feature_engineering(df):
    df['pickup_datetime'] = pandas.to_datetime(df['pickup_datetime'])
    df['weekday'] = df['pickup_datetime'].dt.weekday
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_year'] = df['pickup_datetime'].dt.day + df['pickup_datetime'].dt.month*30
    df['week_of_year'] = df['pickup_datetime'].dt.weekofyear
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['weekhour'] = df['hour'] + df['weekday']*24
    df['haversine'] = haversine_array(df['pickup_latitude'].values,
                                      df['pickup_longitude'].values,
                                      df['dropoff_latitude'].values,
                                      df['dropoff_longitude'].values)
    df['manhatan_dist'] = dummy_manhattan_distance(df['pickup_latitude'].values,
                                                   df['pickup_longitude'].values,
                                                   df['dropoff_latitude'].values,
                                                   df['dropoff_longitude'].values)
    df['bearing'] = bearing_array(df['pickup_latitude'].values,
                                  df['pickup_longitude'].values,
                                  df['dropoff_latitude'].values,
                                  df['dropoff_longitude'].values)
    df['mean_dist'] = (df['manhatan_dist'] + df['haversine'])/2
                                  
    dummys_flag = pandas.get_dummies(df['store_and_fwd_flag'])    
    df = pandas.concat([df,dummys_flag],axis=1)
    df = df.rename(columns={'N': 'flag_N', 'Y': 'flag_Y'})
    df = df.drop('store_and_fwd_flag',1)                                      
    return df
    
train = feature_engineering(train)


test = pandas.read_csv('test.csv')
test = feature_engineering(test)

#scaling data 
# not done yet, maybe not needed?
variables = {'pickup_longitude':'pickup1','pickup_latitude':'pickup2',
 'dropoff_longitude':'dropoff1','dropoff_latitude':'dropoff2',
 'haversine':'haversine_sca','manhatan_dist':'manhatan_sca','bearing':'bearing_sca'}
keys = list(variables.keys())



def scaler(train,test):
   
    
    for i in keys:
        scaler = StandardScaler()
        scaler.fit(train[[i]])
        train[variables[i]] = scaler.transform(train[[i]])
        test[variables[i]] = scaler.transform(test[[i]])
    return train,test
        
train,test = scaler(train,test)        
    
    
def clusters(train,test):
    coords = numpy.vstack((train[['pickup1', 'pickup2']].values,
                    train[['dropoff1', 'dropoff2']].values))
    sample_ind = numpy.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=150, batch_size=10000).fit(coords[sample_ind])
    train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup1', 'pickup2']])
    train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff1', 'dropoff2']])
    test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup1', 'pickup2']])
    test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff1', 'dropoff2']])
    return train,test


train,test = clusters(train,test)                    
    
    
#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corrmat, vmax=.8, square=True);


#trip duration correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'trip_duration')['trip_duration'].index
cm = numpy.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
 


#trining 
train['log_trip_duration'] = numpy.log(train['trip_duration'].values + 1)

train = train.drop(['id','pickup_datetime','dropoff_datetime','trip_duration'],axis=1)
train = train.drop(keys,axis=1)


train, valid = train_test_split(train ,test_size = 0.2)

train_X = train.drop('log_trip_duration',axis=1)
train_Y = train['log_trip_duration']


test_X = valid.drop('log_trip_duration',axis=1)
test_Y = valid['log_trip_duration']

dtrain = xgb.DMatrix(train_X, label=train_Y)
dvalid = xgb.DMatrix(test_X, label=test_Y)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


# run to get the best parameters

#max_depth = [6,20]
#learning_rate = [0.1,0.5,0.7]
#min_child_weight = [20,25,30]
#n_estimators=[30,50,1000]
#best ={'depth':0,'lr':0,'mcw':0,'score':9999}
#for ne in n_estimators:
#    for m in max_depth:
#        for l in learning_rate:
#            for n in min_child_weight:
#                print ('{} Min Child Weight,{} Learning Rate, {} Max_depth'.format(n,l,m))
#                t0 = datetime.now()
#                xgb_pars = {'min_child_weight': n, 'eta': l, 'colsample_bytree': 0.9, 
#                            'max_depth': m,
#                'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
#                'eval_metric': 'rmse', 'objective': 'reg:linear','n_estimators ': ne}
#                model = xgb.train(xgb_pars, dtrain, 50, watchlist, early_stopping_rounds=10,
#                      maximize=False, verbose_eval=1)
#                if model.best_score < best['score']:
#                    best['score'] = model.best_score
#                    best['depth'] = m
#                    best['lr'] = l
#                    best['mcw'] = n
 
xgb_pars = {'min_child_weight': 20, 'eta': 0.1, 'colsample_bytree': 0.9, 
            'max_depth': 20,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear','n_estimators': 1000}
model = xgb.train(xgb_pars, dtrain, 100, watchlist, early_stopping_rounds=10,
      maximize=False, verbose_eval=1) 


xgb.plot_importance(model,)

test = test.drop(keys,axis=1)
test = test.drop(['id','pickup_datetime'],axis=1)

dtest = xgb.DMatrix(test)

 
pred = model.predict(dtest)
pred = numpy.exp(pred) - 1

testid = pandas.read_csv('test.csv')
testid = testid['id']
testid = pandas.concat([testid,pandas.DataFrame(pred)],axis=1)
testid.columns = ['id','trip_duration']
testid.to_csv("submission.csv", index=False)



