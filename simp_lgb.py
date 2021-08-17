import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import log_loss
import lightgbm as lgb
import  time
import os
from tqdm import tqdm
train_path = r"D:\Datasets\GISCUP21\giscup_2021\giscup_2021\train_fea"
test_path = r"D:\Datasets\GISCUP21\giscup_2021\giscup_2021\test_fea\20200901.csv"
lk_path = "./link2vec/linkfea"
test_lk_path = r"D:\Datasets\GISCUP21\giscup_2021\link2vec\linkfea\g_20200901.csv"
submission_path = r"D:\Datasets\GISCUP21\giscup_2021\sample_submission.csv"
weather_path = r"D:\Datasets\GISCUP21\giscup_2021\giscup_2021\weather.csv"

submission = pd.read_csv(submission_path)
weather = pd.read_csv(weather_path)
weather_mapping = {'rainstorm': 5,'heavy rain': 4,'moderate rain': 3,'showers': 2,'cloudy': 1}
weather['weather'] = weather['weather'].map(weather_mapping)
print(weather.head())


# MAPE和SMAPE需要自己实现
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def read_df(path = test_path, date = 20200901, ):
    df_test =  pd.read_table(path,header=None,sep=';;',names=['head','link','cross'])

    # 取head部分
    df_test_head = df_test['head'].str.split(' ',expand = True)
    df_test_head.columns =  ["order_id","ata","distance","simple_eta","driver_id","slice_id"]
    df_test_head[['ata','distance',"simple_eta"]] = df_test_head[['ata','distance',"simple_eta"]].astype(float)
    df_test_head[["driver_id","slice_id"]] = df_test_head[["driver_id","slice_id"]].astype(int)
    df_test_head['weather'] = weather[weather['date']==date]['weather'].values[0]
    df_test_head['hightemp'] = weather[weather['date']==date]['hightemp'].values[0]
    df_test_head['lowtemp'] = weather[weather['date']==date]['lowtemp'].values[0]
    # print(df_test_head.head())

    # 取link部分
    # df_test_link = df_test['link'].str.split(' ',expand = True) # n个类似于261397:0.7493,0.1710,0,0的数据
    # df_test_link.columns =  ["link_id","link_time","link_ratio","link_current_status","link_arrival_status"]
    # print(df_test_link)

    # 取cross部分
    # df_test_cross = df_test['cross'].str.split(' ',expand = True) # m个类似于471304_105381:24的数据
    # df_test_cross.columns = ["cross_id","cross_time"]
    # print(df_test_cross)
    return df_test_head


# df_test_head = read_df()
df_test_head =  pd.read_csv(test_path)
df_test_lk_head =  pd.read_csv(test_lk_path)
del df_test_lk_head['type']
df_test_head = pd.concat([df_test_head,df_test_lk_head],axis=1)
df_test_head['d_s'] = df_test_head['distance']*df_test_head['slice_id']
print(df_test_head.head())

filenames = os.listdir(train_path)
filenames.sort(key=lambda x: int(x[6:8]))
train = []
# filenames.remove("20200803.txt") # 文件为空
filenames = filenames[:2]
for file in tqdm(filenames):
    date = int(file[0:8])
    txt_path = train_path + '/' + file
    # df_head = read_df(txt_path,date)
    df_head = pd.read_csv(txt_path)
    pth = lk_path + '/g_' + file
    df_lk_head = pd.read_csv(pth)
    del df_lk_head['type']
    df_head = pd.concat([df_head, df_lk_head], axis=1)
    df_head['d_s'] = df_head['distance'] * df_head['slice_id']
    train.append(df_head)
df_train_head = pd.concat(train)

param = {'num_leaves': 256,
         'min_data_in_leaf': 60,
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8,
         "bagging_seed": 11,
         "metric": 'mape',}

max_iter = 1
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(df_train_head))
categorical_columns = ['weather',"driver_id","slice_id"]
features = ["distance","simple_eta","driver_id","slice_id", 'weather', 'hightemp', 'lowtemp', 'true_t', 'wight_t','d_s']
lk_fea = ["x_" + str(i) for i in range(32)]
features = features + lk_fea
label = ['ata']
predictions = np.zeros(len(df_test_head))

start = time.time()
feature_importance_df = pd.DataFrame()
start_time = time.time()
score = [0 for _ in range(folds.n_splits)]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train_head.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(df_train_head.iloc[trn_idx][features],
                           label=df_train_head.iloc[trn_idx][label],
                           categorical_feature=categorical_columns)
    val_data = lgb.Dataset(df_train_head.iloc[val_idx][features],
                           label=df_train_head.iloc[val_idx][label],
                           categorical_feature=categorical_columns)

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds=200)

    oof[val_idx] = clf.predict(df_train_head.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
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
        current_pred[idx] = clf.predict(df_test_head.iloc[idx][features], num_iteration=clf.best_iteration)
        initial_idx = final_idx
    predictions += current_pred / min(folds.n_splits, max_iter)

    print("time elapsed: {:<5.2}s".format((time.time() - start_time) / 3600))
    score[fold_] = mape(np.squeeze(df_train_head.iloc[val_idx][label].T.values), oof[val_idx])
    print("*"*10,"flod:",fold_," mape:", score[fold_],"*"*10)
    if fold_ == max_iter - 1: break

if (folds.n_splits == max_iter):
    print("CV score: {:<8.5f}".format(mape(np.squeeze(df_train_head[label].T.values), oof)))
else:
    print("CV score: {:<8.5f}".format(sum(score) / max_iter))

submission['result'] = predictions
submission.to_csv('subs/simple_lgb_sub.csv',index=False)