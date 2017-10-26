import pandas as pd
import numpy as np
import xgboost as xgb
#import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc

# custom objective function (similar to auc)

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True

base_path = '/home/debnomics/porto/' # your folder
print('loading files...')
dt_train = pd.read_csv(base_path+'data/train.csv')
dt_test = pd.read_csv(base_path+'data/test.csv')
trainID = dt_train['id'].values
testID = dt_test['id'].values
dt_test.loc[:,'target']=0
dt_data=pd.concat([dt_train,dt_test])
del dt_test
del dt_train

#Create Folds and Random numbers
seq=list(range(1,6,1))
times=[int(dt_data.shape[0]/5)+1][0]
fold5=(seq*times)[0:dt_data.shape[0]]
dt_data.loc[:,'fold5']=fold5
del_col = list(['id','target','fold5'])

cat_vars=list(dt_data.columns[dt_data.columns.str.endswith('_cat')])
cat_vars.remove('ps_car_11_cat')
bin_vars=list(dt_data.columns[dt_data.columns.str.endswith('_bin')])
oth_vars=list(['id','target','ps_car_11_cat','fold5'])
ord_vars=list(set(list(dt_data.columns))-set(list(cat_vars)+list(bin_vars)+list(oth_vars)))
#create factor for categorical variables
#Pending..

xgb_vars=list(set(list(dt_data.columns))-set(oth_vars))

trainRows=len(trainID)
testRows=len(testID)

x_train=dt_data[xgb_vars][0:trainRows-1]
y_train=dt_data['target'][0:trainRows-1]
x_test=dt_data[xgb_vars][trainRows:]
sub=test['id'].to_frame()
sub['target']=0
# xgb
params = {'eta': 0.5,
'max_depth': 4,
'subsample': 0.9,
'colsample_bytree': 0.9,
'objective': 'binary:logistic',
'eval_metric': 'auc',
'silent': True,
"print.every.n": 1}

for i in list([1,2,3,4,5]):
    print(i)
    fold_index=dt_data[0:trainRows-1].fold5==i
    X_train, X_valid = x_train[-fold_index],x_train[fold_index]
    Y_train, Y_valid = y_train[-fold_index],y_train[fold_index]
    d_train = xgb.DMatrix(X_train, Y_train)
    d_valid = xgb.DMatrix(X_valid, Y_valid)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    xgb_model = xgb.train(params, d_train, 300, watchlist, early_stopping_rounds=25,
                          feval=gini_xgb, maximize=True, verbose_eval=10)
