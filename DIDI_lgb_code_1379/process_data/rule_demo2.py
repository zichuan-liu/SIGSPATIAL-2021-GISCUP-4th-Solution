import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import log_loss
from tqdm import tqdm
import  time
import os
train_path = r"../data/train/"
test_path = r"../data/test/20200901_test.txt"
submission_path = r"../data/sample_submission.csv"
weather_path = r"../data/weather.csv"
link_path = r"../data/nextlinks.txt"

submission = pd.read_csv(submission_path)
link_id = pd.read_table(link_path, header=None,sep=' ')
link_id.columns = ["link_id", "next_link_id"]
link_id[['link_id']] = link_id[['link_id']].astype(int)
map1 = {mod: i + 1 for i, mod in enumerate(link_id['link_id'].unique())}

weather = pd.read_csv(weather_path)
weather_mapping = {'rainstorm': 5,'heavy rain': 4,'moderate rain': 3,'showers': 2,'cloudy': 1}
weather['weather'] = weather['weather'].map(weather_mapping)
all_topo_link = {}
all_topo_cross = {}


def get_link_fea(links, crosses):
    maps = []

    links = list(filter(None, links))
    crosses = list(filter(None, crosses))
    sum_link_time = 0
    sum_link_time_w = 0
    sum_stut = 0
    link_nums = len(all_topo_link)
    link_states_start = [0, 0, 0, 0] #统计出发时四种状态的占比
    link_states_true = [0, 0, 0, 0] #统计到达该路段的四种状态占比 （预测四分类做stacking，或者统计偏差（与天气做二阶特征））
    state_sum_time = [0,0,0,0]
    for link in links:
        mp = []
        link_fe = link.split(',')
        link_id =int(link_fe[0].split(':')[0])
        if link_id not in all_topo_link.keys():
            link_nums = link_nums + 1
            all_topo_link[link_id] = link_nums
            mp.append(link_nums) #记录每个link重新编码后的id
        else:
            mp.append(all_topo_link[link_id])
        #mp.append(int(link_fe[0].split(':')[0])) #没有保存linkid
        mp.append(float(link_fe[0].split(':')[1]))
        mp.append(float(link_fe[1]))
        mp.append(int(link_fe[2])+1) #为所有路口状态+1，去除零值
        mp = np.array(mp)
        sum_link_time += mp[1]#*mp[2] # 所有link的时间和
        sum_link_time_w += mp[1]*(1 + mp[3]/(4+3+2+1))#*mp[2] #算一下同行时间*路况权重（堵4-通2，1为状态未知）
        # 出发时收集到的信息
        if mp[3] == 1:  #可以根据link的上下游信息预测当前路口的状态（但是可能存在连续为0的情况）
            link_states_start[0] +=1
            state_sum_time[0] += mp[1] # 未知路段所花费时间
        elif mp[3] == 2:
            link_states_start[1] += 1
            state_sum_time[1] += mp[1]
        elif mp[3] == 3:
            link_states_start[2] += 1
            state_sum_time[2] += mp[1]
        else:
            link_states_start[3] += 1
            state_sum_time[3] += mp[1]
        sum_stut += mp[3] #当前link的状态，统计所有link的状态和（这里可能有点问题，全是未知的状态反而该项系数最小，用均值替换似乎合理一点）
        # 到达时刻的偏差（穿越特征） 很多都为0
        #print(int(link_fe[3]))
        # 真实到达时的信息
        if int(link_fe[3]) == 1:
            link_states_true[1] +=1
        elif int(link_fe[3]) == 2:
            link_states_true[2] += 1
        elif int(link_fe[3]) == 3:
            link_states_true[3] += 1
        else:
            link_states_true[0] += 1

        maps.append(mp)
    maps = np.array(maps,dtype=np.float32)
    sum_cross_time = 0

    try:
        maps2 = []
        cross_nums = len(all_topo_cross)
        for cross in crosses:
            mp2 = []
            cross_fe = cross.split(':')
            sum_cross_time += int(cross_fe[1]) #考虑十字路口的时间
            cross_id = int(cross_fe[0])
            if cross_id not in all_topo_cross.keys():
                cross_nums = cross_nums + 1
                all_topo_cross[cross_id] = cross_nums
                mp2.append(cross_nums)  # 记录每个link出现的次数
            else:
                mp2.append(all_topo_cross[cross_id])
            mp2.append(int(cross_fe[1]))
            maps2.append(mp2)
        maps2 = np.array(maps2, dtype=np.float32)  # maps2会存在空值[]
        return maps, sum_link_time, sum_link_time_w, sum_cross_time, sum_stut, maps2, link_states_start, link_states_true, state_sum_time

    except:
        maps2 =[]
        mp2=[]
        mp2.append(int(0))
        mp2.append(int(0))
        maps2.append(mp2)
        maps2 = np.array(maps2, dtype=np.float32)
        return maps, sum_link_time, sum_link_time_w, sum_cross_time, sum_stut, maps2, link_states_start, link_states_true, state_sum_time

    
    
def stack_padding(it):
    max_len = max([len(arr) for arr in it])
    padded = np.array([np.lib.pad(arr, ((max_len - len(arr),0),(0,0)), 'constant', constant_values=0) for arr in it],dtype=np.float32)
    return padded

def read_df(path = test_path, date = 20200901, ):
    df_test =  pd.read_table(path,header=None,sep=';;',names=['head','link','cross'])
    ## head
    df_test_head = df_test['head'].str.split(' ',expand = True)
    df_test_head.columns =  ["order_id","ata","distance","simple_eta","driver_id","slice_id"]
    df_test_head['weather'] = weather[weather['date']==date]['weather'].values[0]
    df_test_head['hightemp'] = weather[weather['date']==date]['hightemp'].values[0]
    df_test_head['lowtemp'] = weather[weather['date']==date]['lowtemp'].values[0]
    # print(df_test_head.head())

    ## link
    df_test_link = df_test['link'].str.split(' ',expand = True) # n 261397:0.7493,0.1710,0,0
    # df_test_link.columns =  ["link_id","link_time","link_ratio","link_current_status","link_arrival_status"]
    # print(df_test_link)

    ## cross
    df_test_cross = df_test['cross'].str.split(' ',expand = True) # m 471304_105381:24
    # df_test_cross.columns = ["cross_id","cross_time"]
    # print(df_test_cross)
    return df_test_head, df_test_link, df_test_cross

df_test_head, df_test_link, df_test_cross = read_df()
df_test_head['true_t'] = 0.0
df_test_head['wight_t'] = 0.0
df_test_head['cross_t'] = 0.0
df_test_head['status_t'] = 0
df_test_head['link_num'] = 0
df_test_head['cross_num'] = 0
df_test_head['state_num_0'] = 0
df_test_head['state_num_1'] = 0
df_test_head['state_num_2'] = 0
df_test_head['state_num_3'] = 0
df_test_head['state_sum_time_0']= 0  #
df_test_head['state_sum_time_1']= 0  #
df_test_head['state_sum_time_2']= 0  #
df_test_head['state_sum_time_3']= 0  #
# cols = ['slice_id','weather']
# group = df_test_head.groupby(['slice_id','weather']).agg({'simple_eta': ['mean', 'median']})
# group.columns = ['slice_eta_mean','slice_eta_median']
# group.reset_index(inplace=True)
# df_test_head = pd.merge(df_test_head, group, on=cols, how='left')
# del group
# print(df_test_head[df_test_head.isnull().values==True].head())
test_links = []
test_crosses = []
for i in tqdm(range(len(df_test_head))):
    link, t_t, w_t, c_t, s_t , cross, link_states_start, link_states_true, state_sum_time = get_link_fea(df_test_link.loc[[i]].values[0], df_test_cross.loc[[i]].values[0])
    df_test_head['true_t'][i] = t_t #所有link的时间和
    df_test_head['wight_t'][i] = w_t #通过路口堵塞权重的时间和
    df_test_head['cross_t'][i] = c_t #所有路口的时间和
    df_test_head['status_t'][i] = s_t #所有link的状态和
    df_test_head['link_num'][i] = len(link) #路段的数量
    df_test_head['cross_num'][i] = len(cross)  # 路口的数量
    df_test_head['state_num_0'][i] = link_states_start[0]  #
    df_test_head['state_num_1'][i] = link_states_start[1]  #
    df_test_head['state_num_2'][i] = link_states_start[2]  #
    df_test_head['state_num_3'][i] = link_states_start[3]  #
    df_test_head['state_sum_time_0'][i] = state_sum_time[0]  #
    df_test_head['state_sum_time_1'][i] = state_sum_time[1]  #
    df_test_head['state_sum_time_2'][i] = state_sum_time[2]  #
    df_test_head['state_sum_time_3'][i] = state_sum_time[3]  #
    test_links.append(link)
    test_crosses.append(cross) #还是有bug
print("len(all_topo_link): ",len(all_topo_link)) #将每个link进行labelencode然后保存到topolink中
test_links = stack_padding(test_links)
test_crosses = stack_padding(test_crosses)
print(test_links.shape)
print(df_test_head.head())
isExists = os.path.exists("../data/test_fea")
if not isExists:
    os.makedirs("../data/test_fea")
df_test_head.to_csv('../data/test_fea/' + str(20200901) + '.csv', index=False)
np.savez_compressed("../data/test_fea/test_links.npz", data=test_links)
np.savez_compressed("../data/test_fea/test_cross.npz", data=test_crosses)

del test_links
del test_crosses
#

isExists = os.path.exists("../data/train_fea")
if not isExists:
    os.makedirs("../data/train_fea")

filenames = os.listdir(train_path)
filenames.sort(key=lambda x: int(x[6:8]))
train = []
try:
    filenames.remove("20200803.txt") # file null
except:
    pass
filenames = ["20200801.txt","20200802.txt"]
for file in tqdm(filenames):
    date = int(file[0:8])
    txt_path = train_path + '/' + file
    df_train_head, df_train_link, df_train_cross = read_df(txt_path, date)
    df_train_head['true_t'] = 0.0
    df_train_head['wight_t'] = 0.0
    df_train_head['cross_t'] = 0.0
    df_train_head['status_t'] = 0
    df_train_head['link_num'] = 0
    df_train_head['cross_num'] = 0
    df_train_head['state_num_0'] = 0
    df_train_head['state_num_1'] = 0
    df_train_head['state_num_2'] = 0
    df_train_head['state_num_3'] = 0
    df_train_head['state_sum_time_0'] = 0  #
    df_train_head['state_sum_time_1'] = 0  #
    df_train_head['state_sum_time_2'] = 0  #
    df_train_head['state_sum_time_3'] = 0  #
    links = []
    crosses =[]
    for i in range(len(df_train_head)):
        link, t_t, w_t, c_t, s_t, cross, link_states_start, link_states_true, state_sum_time = get_link_fea(df_train_link.loc[[i]].values[0], df_train_cross.loc[[i]].values[0])
        df_train_head['true_t'][i] = t_t
        df_train_head['wight_t'][i] = w_t
        df_train_head['cross_t'][i] = c_t
        df_train_head['status_t'][i] = s_t
        df_train_head['link_num'][i] = len(link)
        df_train_head['cross_num'][i] = len(cross)
        df_train_head['state_num_0'][i] = link_states_start[0]  #
        df_train_head['state_num_1'][i] = link_states_start[1]  #
        df_train_head['state_num_2'][i] = link_states_start[2]  #
        df_train_head['state_num_3'][i] = link_states_start[3]  #
        df_train_head['state_sum_time_0'][i] = state_sum_time[0]  #
        df_train_head['state_sum_time_1'][i] = state_sum_time[1]  #
        df_train_head['state_sum_time_2'][i] = state_sum_time[2]  #
        df_train_head['state_sum_time_3'][i] = state_sum_time[3]  #
        links.append(link)
        crosses.append(cross)
    print("len(all_topo_link): ",len(all_topo_link))
    links = stack_padding(links)
    crosses = stack_padding(crosses)
    df_train_head.to_csv('../data/train_fea/'+str(date)+'.csv',index=False)
    print(date, links.shape)
    np.savez_compressed("../data/train_fea/"+str(date)+"_links.npz", data=links)
    np.savez_compressed("../data/train_fea/" + str(date) + "_crosss.npz", data=crosses)

#if os.path.exists('map_mcdict.npy'):
#    map_dict=np.load('map_mcdict.npy',allow_pickle=True).item()



np.save('./map_mcdict.npy', all_topo_link)


#simple link time+cross time
submission['result'] = df_test_head['simple_eta']
submission.to_csv('../subs/simple_sub.csv',index=False)




