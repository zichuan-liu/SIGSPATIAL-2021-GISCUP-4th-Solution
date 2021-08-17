import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import lightgbm as lgb
print(lgb.__version__)
import  time
import os
from tqdm import tqdm
train_path1 = r"../data/train_fea/20200801.csv"
train_path2 = r"../data/train_fea/20200802.csv"
test_path = r"../data/test_fea/20200901.csv"
submission_path = r"../sample_submission.csv"
train_path = r"../data/train_fea/"
filenames = os.listdir(train_path)
filenames.sort(key=lambda x: int(x[6:8]))
train_data = []
#filenames = ["20200801.csv","20200802.csv"]
for file in tqdm(filenames):
    try:
        train_data1 = pd.read_csv(train_path+file)
        train_data1['date'] = int(file[6:8])
        if len(train_data)==0:
            train_data = train_data1
        else:
            train_data = pd.concat([train_data, train_data1])
    except:
        pass

print(train_data.shape)
# submission = pd.read_csv(submission_path)
# train_data1 = pd.read_csv(train_path1)
# train_data1['date'] = 1
# train_data2 = pd.read_csv(train_path2)
# train_data2['date'] = 2

#train_data = pd.concat([train_data1,train_data2])
test_data =  pd.read_csv(test_path)
test_data['date'] = 32
data = pd.concat([train_data,test_data]).reset_index(drop=True)
'''
所有特征
order_id,  订单id
ata,       实际行程时间（label）
distance,  订单距离
simple_eta,所有路段路口的平均时间和 （true_t+cross_t）
driver_id, 司机id
slice_id,  出发时刻
weather,   天气
hightemp,  最高气温
lowtemp,   最低气温
true_t,    所有link的平均时间和
wight_t,   计算路段通堵情况加权的link时间和
cross_t,   路口平均时间和
status_t,  状态和
link_num   link数量
cross_num  cross数量
'''
#########################################单值扩展特征，包括均值和统计特征############################################
data['hour'] = data['slice_id'].apply(lambda x: x*5//60) #转化为小时
data['week_day'] = data['date'].apply(lambda x: x%7 + 1)
data['Temp_diff'] = data['hightemp']-data['lowtemp'] #温差
data['expect_velocity'] = data['distance']/data['simple_eta'] #预期速度
data['true_t_mean'] = data['true_t']/data['link_num'] # link的平均时间
data['wight_t_mean'] = data['wight_t']/data['link_num'] # 加权的link平均时间
data['status_t_mean'] = data['status_t']/data['link_num'] # 四种状态的平均状态
data['status_t_mean2'] = (data['state_num_1'] + 2*data['state_num_2'] + 3*data['state_num_3'])/3  #三种状态的平均状态(去除未知状态) (全为nan)
data['link_lens'] = data['distance']/data['link_num'] # link的平均长度
data['cross_t_mean'] = data['cross_t']/data['cross_num'] # 路口的平均通行时间


#比率特征
#state_ratio(全为nan)
data['status_0_ratio'] = data['state_num_0']/(data['state_num_0'] + data['state_num_1'] + data['state_num_2'] + data['state_num_3'])
data['status_1_ratio'] = data['state_num_1']/(data['state_num_0'] + data['state_num_1'] + data['state_num_2'] + data['state_num_3'])
data['status_2_ratio'] = data['state_num_2']/(data['state_num_0'] + data['state_num_1'] + data['state_num_2'] + data['state_num_3'])
data['status_3_ratio'] = data['state_num_3']/(data['state_num_0'] + data['state_num_1'] + data['state_num_2'] + data['state_num_3'])

# 不同路段每种状态下的平均通行时间
data['state_mean_time_0'] = data['state_sum_time_0']/data['state_num_0']
data['state_mean_time_1'] = data['state_sum_time_1']/data['state_num_1']
data['state_mean_time_2'] = data['state_sum_time_2']/data['state_num_2']
data['state_mean_time_3'] = data['state_sum_time_3']/data['state_num_3']
#time_ratio


#时序特征
data["num_order_in_weather"] = data.groupby(["weather"])["order_id"].transform('count')
data["weather_nums"] = data.groupby(["weather"])["date"].transform('nunique')
data["order_nums_in_each_weather"] = data["num_order_in_weather"]/(data["weather_nums"])

#cross_link_ratio 全量直接提升5k?????
data['cross_t_ratio'] = data['cross_t']/data['simple_eta'] #cross时间占比
data['link_t_ratio'] = data['true_t']/data['simple_eta'] #link时间占比
# data['cross_link_ratio'] = data['true_t']/data['cross_t'] #cross/link时间占比
# data['cross_link_num_ratio'] = data['link_num']/data['cross_num'] #cross/link数量比
data['cross_num_ratio'] = data['cross_num']/(data['cross_num']+data['link_num']) #cross占比
data['link_num_ratio'] = data['link_num']/(data['cross_num']+data['link_num']) #link占比
#########################################与司机有关的聚合特征###############################################
#过去三天的司机接单数量
ns = [3,5,7]
start_day=1
end_day=32
print('==>generate statis feature: 统计过去n天司机的接单数量')
for day in ns:
    name = f"driverid_sum_day{day}_count"

    features = data[["driver_id", "order_id", "date"]]
    res_arr = []
    for start in range(start_day, end_day - day + 1):
        temp = features[(features["date"]) >= start & (features["date"] < (start + day))]
        temp = temp.groupby(["driver_id"])["order_id"].count().reset_index()
        temp = temp.rename(columns={"order_id": name})
        temp["date"] = start + day
        res_arr.append(temp)
    features = pd.concat(res_arr)
    features = features[["driver_id", "date", name]]

    data = data.merge(features,how='left',on=["driver_id","date"])

print('==>generate statis feature: 过去n天司机接单的时间段数量')
for day in ns:
    name = f"driverid_hour_sum_day{day}_count"

    features = data[["driver_id", "hour", "date"]]
    res_arr = []
    for start in range(start_day, end_day - day + 1):
        temp = features[(features["date"]) >= start & (features["date"] < (start + day))]
        temp = temp.groupby(["driver_id"])["hour"].count().reset_index()
        temp = temp.rename(columns={"hour": name})
        temp["date"] = start + day
        res_arr.append(temp)
    features = pd.concat(res_arr)
    features = features[["driver_id", "date", name]]

    data = data.merge(features,how='left',on=["driver_id","date"])

#前1天每个时段司机的数量
print('==>generate statis feature: 每小时接单的司机数量')
for day in [1]:
    name = f"time_slot_driver_num_day{day}_count"

    features = data[["driver_id", "hour", "slice_id","date"]]
    res_arr = []
    for start in range(start_day, end_day - day + 1):
        temp = features[(features["date"]) >= start & (features["date"] < (start + day))]
        temp = temp.groupby(["hour"])["driver_id"].count().reset_index()
        temp = temp.rename(columns={"driver_id": name})
        temp["date"] = start + day
        res_arr.append(temp)
    features = pd.concat(res_arr)
    features = features[["hour", "date", name]]

    data = data.merge(features,how='left',on=["hour","date"])
print('==>generate statis feature: 每5分钟接单的司机数量数量')
for day in [1]:
    name = f"time_slot_5minit_driver_num_day{day}_count"

    features = data[["driver_id", "hour", "slice_id","date"]]
    res_arr = []
    for start in range(start_day, end_day - day + 1):
        temp = features[(features["date"]) >= start & (features["date"] < (start + day))]
        temp = temp.groupby(["slice_id"])["driver_id"].count().reset_index()
        temp = temp.rename(columns={"driver_id": name})
        temp["date"] = start + day
        res_arr.append(temp)
    features = pd.concat(res_arr)
    features = features[["slice_id", "date", name]]

    data = data.merge(features,how='left',on=["slice_id","date"])

#########################################合并w2v特征############################################
print('==> add w2v feature')
lkid_vecs = pd.read_pickle(r'../data/w2v_fea/all_day_link_vecs.pkl')  # , nrows = 4
lkid_vecs = lkid_vecs.reset_index(drop=True)
data = data.reset_index(drop=True)
print(lkid_vecs.shape,data.shape)
new_data = pd.concat([data,lkid_vecs], axis=1) #直接和data进行拼接
del lkid_vecs
new_data = new_data.reset_index(drop=True)

croid_vecs = pd.read_pickle(r'../data/w2v_fea/all_day_ctoid_vecs.pkl')  # , nrows = 4
croid_vecs = croid_vecs.reset_index(drop=True)
new_data = new_data.reset_index(drop=True)
print(croid_vecs.shape,new_data.shape)
new_data = pd.concat([new_data,croid_vecs], axis=1) #直接和data进行拼接
del croid_vecs
new_data = new_data.reset_index(drop=True)
print(new_data.shape)

curent_state = pd.read_pickle(r'../data/w2v_fea/all_state_vecs.pkl')  # , nrows = 4
curent_state = curent_state.reset_index(drop=True)
new_data = new_data.reset_index(drop=True)
print(curent_state.shape,new_data.shape)
new_data = pd.concat([new_data,curent_state], axis=1) #直接和data进行拼接
del curent_state
new_data = new_data.reset_index(drop=True)
print(new_data.shape)


# 保存为pkl文件
train_data = new_data[~(new_data['date'] == 32)]
test_data = new_data[new_data['date'] == 32]
print('train_data:',train_data.shape)
print('test_data:',test_data.shape)
train_data.to_pickle("../data/data_train_feature.pkl")
test_data.to_pickle("../data/data_test_feature.pkl")
print('finished')
