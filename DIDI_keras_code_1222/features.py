import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import  time
import os

train_path = r"./giscup_2021/train_fea/"
test_path = r"./giscup_2021/test_fea/20200901.csv"
submission_path = r"./sample_submission.csv"
weather_path = r"./giscup_2021/weather.csv"
lk_fea_names = ['time']#, 'status','ratio'
lk_path = "./link2vec/linkfea_"
test_lk_path = r"g_20200901.csv"
features = ["distance","simple_eta","driver_id","slice_id", 'weather', 'hightemp', 'lowtemp']

def get_his_features(data1,keys,values):
    data = data1.copy()
    str_key = ""
    for k in keys:
        str_key += k
        str_key += "_"
    for v in values:
        str_key1 = str_key + v + "_mean"
        f = data.groupby(keys)[v].mean().reset_index(name=str_key1)
        data = pd.merge(data, f, 'left', on=keys)

    return data

def prepare(data1):
    data = data1.copy()
    
    data['detatemp'] = data['hightemp'] - data['lowtemp']
    data['link_cross_num'] = data['link_num'] + data['cross_num']
    data['speed'] = data['distance']/data['simple_eta']
    data['status_speed'] = data['status_t']*1.0/data['link_num']
    cate_feat = ['weather', 'driver_id', 'slice_id', 'date','detatemp', 'hightemp', 'lowtemp', 'link_num', 'cross_num', 'link_cross_num']

    # time feature
    data['hour'] = data['slice_id'].apply(lambda x: x*5//60)
    cate_feat.append('hour')
    
    data['cross_t_ratio'] = data['cross_t']/data['simple_eta']
    data['link_t_ratio'] = data['true_t']/data['simple_eta']
    data['cross_num_ratio'] = data['cross_num']/data['link_cross_num']
    data['link_num_ratio'] = data['link_num']/data['link_cross_num']
    
    lbl = LabelEncoder()
    for i in cate_feat:
        data[i] = lbl.fit_transform(data[i].astype(str))

    data['week_day'] = data['date'].apply(lambda x: x%7 + 1)
    cate_feat.append('week_day')

    values = ['simple_eta', 'distance']#cross_t, status_t, 'true_t', 'cross_t'
    # slice_id features
    data = get_his_features(data, ['slice_id'], values,)
    # week_day  features
    data = get_his_features(data, ['week_day'], values,)
    # hour  features
    data = get_his_features(data, ['hour'], values)
    # date slice_id features
    data = get_his_features(data, ['date', 'slice_id'], values)
    # week_day hour features
    data = get_his_features(data, ['week_day', 'hour'], values)
    # week_day slice_id features
    data = get_his_features(data, ['week_day', 'slice_id'], values)
    # week_day link_num features
    #data = get_his_features(data, ['week_day', 'link_num'], values)
    # hour link_num features
    #data = get_his_features(data, ['hour', 'link_num'], values)
    # week_day link_num features
    #data = get_his_features(data, ['week_day', 'hour', 'link_num'], values)
    # slice_id link_num features
    #data = get_his_features(data, ['hour', 'slice_id', 'link_num'], values)

    f = data.groupby(['week_day', 'hour'])['order_id'].count().reset_index(name="week_cnt")
    data = pd.merge(data, f, 'left', on=['week_day', 'hour'])
    f = data.groupby(['date', 'hour'])['order_id'].count().reset_index(name="date_cnt")
    data = pd.merge(data, f, 'left', on=['date', 'hour'])
    f = data.groupby(['date', 'slice_id'])['order_id'].count().reset_index(name="slice_id_cnt")
    data = pd.merge(data, f, 'left', on=['date', 'slice_id'])
    f = data.groupby(['week_day', 'hour'])['order_id'].count().reset_index(name="hour_cnt")
    data = pd.merge(data, f, 'left', on=['week_day', 'hour'])
    f = data.groupby(['week_day', 'slice_id'])['order_id'].count().reset_index(name="week_day_slice_id_cnt")
    data = pd.merge(data, f, 'left', on=['week_day', 'slice_id'])
    f = data.groupby(['hour', 'slice_id'])['order_id'].count().reset_index(name="hour_slice_id_cnt")
    data = pd.merge(data, f, 'left', on=['hour', 'slice_id'])
    # data['d_s'] = data['distance'] * data['slice_id']
    return data,cate_feat

def get_feature_label(train,test):
    #####mean
    f = train.groupby(['slice_id']).ata.mean().reset_index(name='slice_ata_mean')
    train = pd.merge(train, f, 'left', on=['slice_id'])
    test = pd.merge(test, f, 'left', on=['slice_id'])
    f = train.groupby(['hour']).ata.mean().reset_index(name='hour_ata_mean')
    train = pd.merge(train, f, 'left', on=['hour'])
    test = pd.merge(test, f, 'left', on=['hour'])
    f = train.groupby(['week_day']).ata.mean().reset_index(name='week_day_ata_mean')
    train = pd.merge(train, f, 'left', on=['week_day'])
    test = pd.merge(test, f, 'left', on=['week_day'])
    f = train.groupby(['week_day', 'hour']).ata.mean().reset_index(name='week_hour_ata_mean')
    train = pd.merge(train, f, 'left', on=['week_day', 'hour'])
    test = pd.merge(test, f, 'left', on=['week_day', 'hour'])
    f = train.groupby(['week_day', 'slice_id']).ata.mean().reset_index(name='week_slice_id_ata_mean')
    train = pd.merge(train, f, 'left', on=['week_day', 'slice_id'])
    test = pd.merge(test, f, 'left', on=['week_day', 'slice_id'])
    f = train.groupby(['weather']).ata.mean().reset_index(name='weather_ata_mean')
    train = pd.merge(train, f, 'left', on=['weather'])
    test = pd.merge(test, f, 'left', on=['weather'])
    #f = train.groupby(['link_num']).ata.mean().reset_index(name='link_num_ata_mean')
    #train = pd.merge(train, f, 'left', on=['link_num'])
    #test = pd.merge(test, f, 'left', on=['link_num'])

    return train,test
    

##################################################
submission = pd.read_csv(submission_path)
df_test_head =  pd.read_csv(test_path)
df_test_head['date'] = 20200901
for lk_fea_name in lk_fea_names:
    df_test_lk_head =  pd.read_csv(lk_path+lk_fea_name+'/'+test_lk_path)
    del df_test_lk_head['type']
    df_test_lk_head.columns = [lk_fea_name+"_"+str(x) for x in range(64)]
    #print(df_test_lk_head.head())
    df_test_head = pd.concat([df_test_head,df_test_lk_head],axis=1)

print(df_test_head.columns)

filenames = os.listdir(train_path)
filenames = [f for f in filenames if f[-3:] == 'csv']
filenames.sort(key=lambda x: int(x[6:8]))
train = []
for file in tqdm(filenames):
    date = int(file[0:8])
    txt_path = train_path  + file
    print(txt_path)
    df_head = pd.read_csv(txt_path)
    df_head['date'] = date
    for lk_fea_name in lk_fea_names:
        pth = lk_path+lk_fea_name + '/g_' + file
        print(pth)
        df_lk_head = pd.read_csv(pth)
        #print(df_lk_head.head())
        del df_lk_head['type']
        df_lk_head.columns = [lk_fea_name+"_"+str(x) for x in range(64)]
        df_head = pd.concat([df_head, df_lk_head], axis=1)
    
    train.append(df_head)
df_train_head = pd.concat(train)
print(df_train_head.shape)
#######################################################

big_data = pd.concat([df_train_head, df_test_head],axis = 0)
big_data.reset_index(drop = True,inplace = True)
train_index = len(df_train_head)
big_data, cate_feat = prepare(big_data)
df_train_head = big_data[0:train_index]
df_test_head = big_data[train_index:]
df_test_head.reset_index(drop = True,inplace = True)
del big_data
df_train_head,df_test_head = get_feature_label(df_train_head,df_test_head)
print(df_test_head.columns)
temp = df_test_head.isnull().any() 
temp=pd.DataFrame(data={'colname': temp.index,'isnulls':temp.values})
print(temp.loc[temp.isnulls==True,'colname'])
del temp

print(cate_feat)
df_train_head.to_pickle(r'./giscup_2021/giscup_2021/data_train_feature_lkpro_max_max_max.pkl')
df_test_head.to_pickle(r'./giscup_2021/giscup_2021/data_test_feature_lkpro_max_max_max.pkl')

