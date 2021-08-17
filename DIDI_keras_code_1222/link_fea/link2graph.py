import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import log_loss
from tqdm import tqdm
import  time
import json
import os
from param_parser import parameter_parser
train_path = r"../giscup_2021/giscup_2021/train/"
test_path = r"../giscup_2021/giscup_2021/test/20200901_test.txt"
submission_path = r"../giscup_2021/sample_submission.csv"
weather_path = r"../giscup_2021/giscup_2021/weather.csv"

submission = pd.read_csv(submission_path)
weather = pd.read_csv(weather_path)
weather_mapping = {'rainstorm': 5,'heavy rain': 4,'moderate rain': 3,'showers': 2,'cloudy': 1}
weather['weather'] = weather['weather'].map(weather_mapping)
print(weather.head())
args = parameter_parser()

emb_type = args.feature_type #time, ratio, status
print(emb_type)
print(emb_type)
print(emb_type)
print(emb_type)
print(emb_type)
def get_link(links):
    maps = {}
    maps['edges'] = []  # [id1, id2]
    maps['features'] = {}   # id1 , fea
    links = list(filter(None, links))
    len_ = len(links)

    
    if len_>1:
        for i in range(len_-1):
            link_fe = links[i].split(',')
            link_fe2 = links[i+1].split(',')
            if emb_type == 'time':
              features_link = round(float(link_fe[0].split(':')[1]), 2)
            elif emb_type == 'ratio':
              features_link = round(float(link_fe[1]), 2)
            elif emb_type == 'status':
              features_link = int(link_fe[2])
            
            maps['edges'].append([int(link_fe[0].split(':')[0]), int(link_fe2[0].split(':')[0])])
            maps['features'][link_fe[0].split(':')[0]] = features_link#*float(link_fe[1])
            # mp['link_id'] = link_fe[0].split(':')[0] # id
            # mp['link_time'] = float(link_fe[0].split(':')[1]) # time
            # mp['link_ratio'] = float(link_fe[1])   # ratio
            # mp['link_current_status'] = int(link_fe[2])+1   # st
            # mp['link_arrival_status'] = int(link_fe[3])+1   # st
        link_fe = links[len_-1].split(',')
        if emb_type == 'time':
          features_link = round(float(link_fe[0].split(':')[1]), 2)
        elif emb_type == 'ratio':
          features_link = round(float(link_fe[1]), 2)
        elif emb_type == 'status':
          features_link = int(link_fe[2])
        maps['features'][link_fe[0].split(':')[0]] = features_link# * float(link_fe[1])
    else:
        print("************error**************")
    return maps


def read_df(path = test_path, date = 20200901,):
    df_test = pd.read_table(path,header=None,sep=';;',names=['head','link','cross'])
    df_test_head = df_test['head'].str.split(' ',expand = True)
    df_test_head.columns =  ["order_id","ata","distance","simple_eta","driver_id","slice_id"]
    df_test_head['weather'] = weather[weather['date']==date]['weather'].values[0]
    df_test_head['hightemp'] = weather[weather['date']==date]['hightemp'].values[0]
    df_test_head['lowtemp'] = weather[weather['date']==date]['lowtemp'].values[0]
    # print(df_test_head.head())

    df_test_link = df_test['link'].str.split(' ',expand = True)
    # df_test_link.columns =  ["link_id","link_time","link_ratio","link_current_status","link_arrival_status"]
    #print(df_test_link)

    isExists = os.path.exists("./dataset_"+emb_type+"/g_"+str(date))
    if not isExists:
        os.makedirs("./dataset_"+emb_type+"/g_"+str(date))
    for i in tqdm(range(len(df_test_link))):
        maps = get_link(df_test_link.loc[[i]].values[0])
        file_path = "./dataset_"+emb_type+"/g_"+str(date)+"/"+ str(i) +".json"
        with open(file_path, "w") as f:
            f.write(json.dumps(maps, ensure_ascii=False, indent=4, separators=(',', ':')))

    # df_test_cross = df_test['cross'].str.split(' ',expand = True)
    # df_test_cross.columns = ["cross_id","cross_time"]
    # print(df_test_cross)
    return df_test_head, df_test_link#, df_test_cross


read_df()# test

filenames = os.listdir(train_path)
filenames.sort(key=lambda x: int(x[6:8]))
train = []
filenames.remove("20200803.txt")
print(filenames)
for file in tqdm(filenames):
    print(file)
    date = int(file[0:8])
    txt_path = train_path + '/' + file
    read_df(txt_path, date)  # test

