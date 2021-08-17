import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
print(lgb.__version__)
import  time
import os
from tqdm import tqdm
train_path = r"./data/data_train_feature.pkl"
test_path = r"./data/data_test_feature.pkl"
submission_path = r"./data/sample_submission.csv"

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

def recognize_feature(data, categorical_columns=[], lable = ['ata', 'order_id']):
    sparse_features = []
    dense_features = []
    for f in data.columns:
        if f in lable:
            pass
        elif data[f].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
            sparse_features.append(f)
        elif f in categorical_columns:
            lbl = LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
            sparse_features.append(f)
        elif data[f].dtype not in ['float16', 'float32', 'float64']:
            if (len(data[f].unique()) < 1000):
                lbl = LabelEncoder()
                lbl.fit(list(data[f].values))
                data[f] = lbl.transform(list(data[f].values))
                sparse_features.append(f)
    print("sparse : unique sum ", sum([len(data[f].unique()) for f in sparse_features]))

    dense_features = list(set(data.columns.tolist()) - set(sparse_features))
    dense_features = [x for x in dense_features if x not in lable]
    return data, sparse_features, dense_features

categorical_columns = ['weather',"driver_id","slice_id"]

submission = pd.read_csv(submission_path)
df_test_head =  pd.read_pickle(test_path)
df_train_head = pd.read_pickle(train_path)#, nrows = 148457
data = pd.concat([df_train_head, df_test_head],axis = 0)
data.reset_index(drop = True, inplace = True)
train_index = len(df_train_head)
#data['hour'] = data['slice_id'].apply(lambda x: x*5//60)
categorical_columns.append('hour')
#data['week_day'] = data['date'].apply(lambda x: x%7 + 1)
categorical_columns.append('week_day')
data, sparse_features, dense_features = recognize_feature(data, categorical_columns, lable=['order_id','ata', 'date', 'd_s'])

test_feat = ['simple_eta','distance','true_t_mean','expect_velocity','link_num','link_lens','cross_t_mean']
data[test_feat] = data[test_feat].fillna(0, )
data[test_feat] = np.log(data[test_feat] + 1.0)

df_train_head = data[0:train_index]
df_test_head = data[train_index:]
df_test_head.reset_index(drop = True,inplace = True)
del data


print(df_test_head.columns)
print("df_train_head:",df_train_head.shape)
print("df_test_head:",df_test_head.shape)

max_iter = 10
n_splits = 5
folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)
oof = np.zeros(len(df_train_head))
categorical_columns = sparse_features#['weather',"driver_id","slice_id"]
drop_feature = ['state_sum_time_0','state_sum_time_1','state_sum_time_2','state_sum_time_3','state_num_0','state_num_1','state_num_2','state_num_3',
                'state_mean_time_0','state_mean_time_1','state_mean_time_2','state_mean_time_3',]
features = [f for f in df_test_head.columns if f not in ['order_id','ata', 'date']] #非这三项其余全作为特征
label = ['ata']
predictions = np.zeros(len(df_test_head))
print(features)
start = time.time()
feature_importance_df = pd.DataFrame()
start_time = time.time()
score = [0 for _ in range(folds.n_splits)]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train_head.values)):
    print("fold n {}".format(fold_))
    # trn_data = lgb.Dataset(df_train_head.iloc[trn_idx][features],
    #                        label=df_train_head.iloc[trn_idx][label],
    #                        categorical_feature=categorical_columns)
    # val_data = lgb.Dataset(df_train_head.iloc[val_idx][features],
    #                        label=df_train_head.iloc[val_idx][label],
    #                        categorical_feature=categorical_columns)

    num_round = 8000
    clf = LGBMRegressor(
        learning_rate=0.05,
        n_estimators=8000,
        num_leaves=512-1,

        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2021,
        metric='None',

    )

    clf.fit(
        df_train_head.iloc[trn_idx][features],df_train_head.iloc[trn_idx][label],
        eval_set= [(df_train_head.iloc[val_idx][features], df_train_head.iloc[val_idx][label])],#[(df_train_head.iloc[val_idx][features],df_train_head.iloc[val_idx][label])],
        eval_metric='mape',
        early_stopping_rounds=200,
        verbose=100,
        #categorical_feature = categorical_columns
    )


    oof[val_idx] = clf.predict(df_train_head.iloc[val_idx][features], num_iteration=clf.best_iteration_)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    #fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print(feature_importance_df.head(15))

    # we perform predictions by chunks
    initial_idx = 0
    chunk_size = 1000000
    current_pred = np.zeros(len(df_test_head))
    while initial_idx < df_test_head.shape[0]:
        final_idx = min(initial_idx + chunk_size, df_test_head.shape[0])
        idx = range(initial_idx, final_idx)
        current_pred[idx] = clf.predict(df_test_head.iloc[idx][features], num_iteration=clf.best_iteration_)
        initial_idx = final_idx
    predictions += current_pred / min(folds.n_splits, max_iter)

    print("time elapsed: {:<5.2}s".format((time.time() - start_time) / 3600))
    score[fold_] = mape(np.squeeze(df_train_head.iloc[val_idx][label].T.values), oof[val_idx])
    print("*"*10,"flod:",fold_," mape:", score[fold_],"*"*10)
    if fold_ == max_iter - 1: break

if (folds.n_splits == max_iter):
    print("CV score: {:<8.5f}".format(mape(np.squeeze(df_train_head[label].T.values), oof)))
else:
    print("CV score: {:<8.5f}".format(sum(score) /  min(folds.n_splits, max_iter)))

submission['result'] = predictions
submission.to_csv('./subs/sub_%.6f.csv' % (sum(score) /  min(folds.n_splits, max_iter)),index=False)