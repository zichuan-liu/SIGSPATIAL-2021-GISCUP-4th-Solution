import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import log_loss
from tqdm import tqdm
import  time
import os
train_path = r"./giscup_2021/train/"
test_path = r"./giscup_2021/test/20200901_test.txt"
submission_path = r"./sample_submission.csv"
weather_path = r"./giscup_2021/weather.csv"
link_path = r"./giscup_2021/nextlinks.txt"

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
topo_link_id = np.load('./giscup_2021/all_topo_link_dict.npy',allow_pickle=True).item()
next_link = {}
def get_link_fea(links, crosses):  
    links = list(filter(None, links))
    crosses = list(filter(None, crosses))
    maps = []
    sum_link_time = 0
    sum_link_time_w = 0
    sum_stut = 0
    link_nums = len(all_topo_link)
    for link in links:
        mp = []
        link_fe = link.split(',')
        link_id =int(link_fe[0].split(':')[0])
        if link_id not in all_topo_link.keys():
            link_nums = link_nums + 1
            all_topo_link[link_id] = link_nums
            mp.append(link_nums)
        else:
            mp.append(all_topo_link[link_id])
        
        #mp.append(int(link_fe[0].split(':')[0]))
        mp.append(float(link_fe[0].split(':')[1]))
        mp.append(float(link_fe[1]))
        mp.append(int(link_fe[2])+1)
        mp.append(mp[1]*mp[2])  # time_ro
        mp.append(mp[1]*(1 + mp[3]/(4+3+2+1)))# w_time
        
        mp = np.array(mp)
        sum_link_time += mp[1]#*mp[2]
        sum_link_time_w += mp[1]*(1 + mp[3]/(4+3+2+1))#*mp[2]
        sum_stut += mp[3]
        maps.append(mp)
    maps = np.array(maps,dtype=np.float32)
    #get maps, sum_link_time,sum_link_time_w,sum_stut
    #################
    
    cross_maps = []
    sum_cross_time = 0
    cross_nums = len(all_topo_cross)
    try:
        for cross in crosses:
            mp = []
            cross_fe = cross.split(':')
            if cross_fe[0] not in all_topo_cross.keys():
                cross_nums = cross_nums + 1
                all_topo_cross[cross_fe[0]] = cross_nums
                mp.append(cross_nums)
            else:
                mp.append(all_topo_cross[cross_fe[0]])
            
            #lk_id1 = int(cross_fe[0].split('_')[0])
            #lk_id2 = int(cross_fe[0].split('_')[1])
            #mp.append(topo_link_id[lk_id1])
            #mp.append(topo_link_id[lk_id2])
            mp.append(float(cross_fe[1]))
            sum_cross_time += float(cross_fe[1])
            cross_maps.append(mp)
    except:
        cross_maps.append([0,0])
        pass
    maps = np.array(maps,dtype=np.float32)
    #get cross_maps, sum_cross_time
    ################
    return  cross_maps,maps,sum_link_time,sum_link_time_w, sum_cross_time, sum_stut
    
    
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
#cols = ['slice_id','weather']
#group = df_test_head.groupby(['slice_id','weather']).agg({'simple_eta': ['mean', 'median']})
#group.columns = ['slice_eta_mean','slice_eta_median']
#group.reset_index(inplace=True)
#df_test_head = pd.merge(df_test_head, group, on=cols, how='left')          
#del group           
#print(df_test_head[df_test_head.isnull().values==True].head())  
test_links = []
test_crosses = []   
for i in tqdm(range(len(df_test_head))):
    cross_maps, link, t_t, w_t, c_t, s_t = get_link_fea(df_test_link.loc[[i]].values[0], df_test_cross.loc[[i]].values[0])
    df_test_head['true_t'][i] = t_t
    df_test_head['wight_t'][i] = w_t
    df_test_head['cross_t'][i] = c_t
    df_test_head['status_t'][i] = s_t
    df_test_head['link_num'][i] = len(link)
    df_test_head['cross_num'][i] = len(link)
    test_links.append(link)
    test_crosses.append(cross_maps)
print("len(all_topo_link): ",len(all_topo_link))
print("len(all_topo_cross): ",len(all_topo_cross))
test_links = stack_padding(test_links)
test_crosses = stack_padding(test_crosses)
print("test_links.shape: ",test_links.shape)
print("test_crosses.shape: ",test_crosses.shape)
print(df_test_head.head())
isExists = os.path.exists("./giscup_2021/test_fea")
if not isExists:
    os.makedirs("./giscup_2021/test_fea")
df_test_head.to_csv('./giscup_2021/test_fea/' + str(20200901) + '.csv', index=False)
np.savez_compressed("./giscup_2021/test_fea/test_links.npz", data=test_links)
np.savez_compressed("./giscup_2021/test_fea/test_crosses.npz", data=test_crosses)
del test_links,test_crosses


isExists = os.path.exists("./giscup_2021/train_fea")
if not isExists:
    os.makedirs("./giscup_2021/train_fea")

filenames = os.listdir(train_path)
filenames.sort(key=lambda x: int(x[6:8]))
train = []
filenames.remove("20200803.txt") # file null
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
    links = []
    crosses = [] 
    for i in range(len(df_train_head)):
        cross_map, link, t_t, w_t, c_t, s_t = get_link_fea(df_train_link.loc[[i]].values[0], df_train_cross.loc[[i]].values[0])
        df_train_head['true_t'][i] = t_t
        df_train_head['wight_t'][i] = w_t
        df_train_head['cross_t'][i] = c_t
        df_train_head['status_t'][i] = s_t
        df_train_head['link_num'][i] = len(link)
        df_train_head['cross_num'][i] = len(cross_map)
        links.append(link)
        crosses.append(cross_map)
    print("len(all_topo_link): ",len(all_topo_link))
    print("len(all_topo_cross): ",len(all_topo_cross))
    links = stack_padding(links)
    crosses = stack_padding(crosses)
    df_train_head.to_csv('./giscup_2021/train_fea/'+str(date)+'.csv',index=False)
    print("links.shape",date, links.shape)
    np.savez_compressed("./giscup_2021/train_fea/"+str(date)+"_links.npz", data=links)
    
    print("crosses.shape",date, crosses.shape)
    np.savez_compressed("./giscup_2021/train_fea/"+str(date)+"_crosses.npz", data=crosses)

#if os.path.exists('map_mcdict.npy'):
#    map_dict=np.load('map_mcdict.npy',allow_pickle=True).item()



np.save('./giscup_2021/all_topo_link_dict.npy',all_topo_link)
np.save('./giscup_2021/all_topo_cross_dict.npy',all_topo_cross)

# simple link time+cross time
submission['result'] = df_test_head['simple_eta']
submission.to_csv('./subs/simple_sub.csv',index=False)




