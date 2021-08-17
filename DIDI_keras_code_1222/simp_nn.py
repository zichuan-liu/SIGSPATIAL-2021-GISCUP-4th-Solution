import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import time
import os
from tqdm import tqdm
import tensorflow as tf
from DeepCTR.deepctr.models import DeepFM, xDeepFM, DCN
from DeepCTR.deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names
import gc
from model import WDR

import random
N_SEED=42
random.seed(N_SEED)
np.random.seed(N_SEED)
tf.random.set_seed(N_SEED)

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1,2,3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
tf.keras.backend.clear_session()
strategy = tf.distribute.MirroredStrategy()

train_rnn_path = r"./giscup_2021/train_fea"
test_rnn_path = r"./giscup_2021/test_fea"
train_path = r"./giscup_2021/data_train_feature_lkpro_max_max_max.pkl"###
test_path = r"./giscup_2021/data_test_feature_lkpro_max_max_max.pkl"###
submission_path = r"./sample_submission.csv"

all_topo_link=np.load('./giscup_2021/all_topo_link_dict.npy',allow_pickle=True).item()
link_emb_size = len(all_topo_link)
del all_topo_link
print("get link_emb_size size: ", link_emb_size)

all_topo_cross=np.load('./giscup_2021/all_topo_cross_dict.npy',allow_pickle=True).item()
cross_emb_size = len(all_topo_cross)
del all_topo_cross
print("get cross_emb_size size: ", cross_emb_size)

lk_fea_names = ['t','ratio','status']
#######################################
links_size = 180
crosses_size = 10
hidden_layer, conv_layer, rnn_dmodel = [512, 128], [128,64], 32
emb_add = False#False
max_iter = 5
n_splits = 5
n_epoch = 15
n_batch = 2048*2
patience = 3
LR = 0.0015
min_LR=0.0002
is_Test = False
is_RNN = True
is_w2v = True
is_save_w2v = False
is_tfidf = False
is_save_tfidf = False#False
is_w2v_cross = False
categorical_columns = ['weather', 'driver_id', 'slice_id', 'date','detatemp', 'hightemp', 'lowtemp', 'link_num', 'cross_num', 'hour', 'week_day', 'link_cross_num']#['weather',"driver_id",'slice_id']

#######################################
label = ['ata']
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

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

submission = pd.read_csv(submission_path)
df_test_head =  pd.read_pickle(test_path)

temp = df_test_head.isnull().any() 
temp=pd.DataFrame(data={'colname': temp.index,'isnulls':temp.values})
print(temp.loc[temp.isnulls==True,'colname'])
del temp

if is_Test:
    df_train_head = pd.read_pickle(train_path)#, nrows = 148457
    df_train_head = df_train_head.head(148457)
else:
    df_train_head = pd.read_pickle(train_path)#, nrows = 148457
    

print(df_train_head.head())
print(df_train_head.shape)
data = pd.concat([df_train_head, df_test_head],axis = 0)
data.reset_index(drop = True, inplace = True)
#data['hour'] = data['slice_id'].apply(lambda x: x*5//60)
#categorical_columns.append('hour')
#data['week_day'] = data['date'].apply(lambda x: x%7 + 1)
#categorical_columns.append('week_day')
train_index = len(df_train_head)


if is_RNN:
    df_test_rnn =  np.load(test_rnn_path+"/test_links.npz")['data'][:, -links_size:,:]
    #print()
    df_test_rnn_crosses =  np.load(test_rnn_path+"/test_crosses.npz")['data'][:, -crosses_size:,:]
    df_test_rnn = df_test_rnn[:,:,:5]################
else:
    df_test_rnn = np.zeros((len(df_test_head),1,1))
    df_test_rnn_crosses = np.zeros((len(df_test_head),1,1))

if is_RNN:
    filenames = os.listdir(train_rnn_path)
    filenames = [f for f in filenames if f[-3:] == 'csv']
    filenames.sort(key=lambda x: int(x[6:8]))
    print(filenames)
    if is_Test:
      filenames = filenames[:2]
    train_rnn = []
    train_rnn_crosses = []
    for file in tqdm(filenames):
        date = int(file[0:8])
        
        txt_path = train_rnn_path + '/' + file[0:8] + '_links.npz'
        df_train_rnn = np.load(txt_path)['data'][:, -links_size:,:]
        df_train_rnn = np.array([np.lib.pad(arr, ((links_size - len(arr) if links_size - len(arr)>0 else 0, 0),(0, 0)), 'constant', constant_values=0) for arr in df_train_rnn],dtype=np.float32)
        df_train_rnn = df_train_rnn[:,:,:5]##################
        print(df_train_rnn.shape)
        train_rnn.append(df_train_rnn)
        
        txt_path = train_rnn_path + '/' + file[0:8] + '_crosses.npz'
        df_train_rnn_crosses = np.load(txt_path)['data'][:, -crosses_size:,:]
        df_train_rnn_crosses = np.array([np.lib.pad(arr, ((crosses_size - len(arr) if crosses_size - len(arr)>0 else 0, 0),(0, 0)), 'constant', constant_values=0) for arr in df_train_rnn_crosses],dtype=np.float32)
        
        print(df_train_rnn_crosses.shape)
        train_rnn_crosses.append(df_train_rnn_crosses)
    
    train_rnn = np.concatenate(train_rnn)
    train_rnn_crosses = np.concatenate(train_rnn_crosses)
    print(train_rnn.shape)
    print(train_rnn_crosses.shape)
else:
    train_rnn = np.zeros((train_index,1,1))
    train_rnn_crosses = np.zeros((train_index,1,1))

#print(df_test_head.columns)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
def tfidf2vec(data,train_rnn_listnum=0, tfidf_size = 16, is_tfidf=is_tfidf,is_save_tfidf = is_save_tfidf):
    if is_tfidf:
        if not is_save_tfidf:
            lkid_list = train_rnn[:, :, train_rnn_listnum].astype(int).tolist()  # his id
            lkid_list = [str(list(filter(lambda x: x != 0, lkid))).strip('[').strip(']').replace(',', '') for
                         lkid in lkid_list]
    
            lkid_list_test = df_test_rnn[:, :, train_rnn_listnum].astype(int).tolist()  # his id
            lkid_list_test = [
                str(list(filter(lambda x: x != 0, lkid))).strip('[').strip(']').replace(',', '') for lkid in
                lkid_list_test]
    
            lkid_list = lkid_list + lkid_list_test
            del lkid_list_test
    
            print(lkid_list[0])

            if train_rnn_listnum==3:
                  tfidfVec = TfidfVectorizer(
                  analyzer='char',  
                  max_features=tfidf_size,
                  # vocabulary=vocabulary
              )
            else:
                  tfidfVec = TfidfVectorizer(
                  analyzer='word',  
                  max_features=tfidf_size,
                  # vocabulary=vocabulary
              )
            
            tfidf_matrix = tfidfVec.fit_transform(lkid_list)
            tfidf_matrix = pd.DataFrame(tfidf_matrix.todense())
            tfidf_matrix.columns = ["tfidf_fea_" + str(train_rnn_listnum) + '_' + str(x) for x in range(tfidf_size)]
            tfidf_matrix.to_pickle(
                    r'./giscup_2021/giscup_2021/tfidf_matrix_fea_' + str(train_rnn_listnum) + '.pkl')
            if is_Test:
                tfidf_matrix = tfidf_matrix.head(148457)
            print( str(train_rnn_listnum), tfidf_matrix.shape)
            data = pd.concat([tfidf_matrix, data], axis=1)
            del tfidf_matrix
            gc.collect()
        else:
            tfidf_matrix = pd.read_pickle(r'./giscup_2021/giscup_2021/tfidf_matrix_fea_' + str(train_rnn_listnum) + '.pkl')  # , nrows = 148457
            if is_Test:
                tfidf_matrix = tfidf_matrix.head(148457)
            
            print( str(train_rnn_listnum), tfidf_matrix.shape)
            data = pd.concat([tfidf_matrix, data], axis=1)
            del tfidf_matrix
            gc.collect()
    else:
        pass
    return data
    
data = tfidf2vec(data,train_rnn_listnum=0, tfidf_size = 32, is_tfidf=is_tfidf,is_save_tfidf = is_save_tfidf)
    
#data = tfidf2vec(data,train_rnn_listnum=3, tfidf_size = 4, is_tfidf=is_tfidf,is_save_tfidf = is_save_tfidf)

from gensim import models
def to_text_vector(words, model):

    #words = txt.split(',')
    array = np.asarray([model.wv[w] for w in words if w in words],dtype='float32')   
    return array.mean(axis=0)
#test
#sentences = ["18,2,3,5",'3,4,18','1,4,2']
#sentences = [lkid.strip('[').strip(']').replace(' ', '').split(',') for lkid in sentences]
#model = models.Word2Vec(sentences, workers=8, min_count = 0,  size = 10, window = 2)
#print(to_text_vector(txt="18,2,3,5", model= model))
#print(model.wv.vocab)
def w2v_fea(data, train_rnn_listnum=0, w2v_size=32, is_w2v=is_w2v, is_save_w2v=is_save_w2v):
    if is_w2v:
        if not is_save_w2v:
            lkid_list = train_rnn[:, :, train_rnn_listnum].astype(int).tolist()  # his id
            lkid_list = [str(list(filter(lambda x: x != 0, lkid))).strip('[').strip(']').replace(' ', '').split(',') for
                         lkid in lkid_list]
    
            lkid_list_test = df_test_rnn[:, :, train_rnn_listnum].astype(int).tolist()  # his id
            lkid_list_test = [
                str(list(filter(lambda x: x != 0, lkid))).strip('[').strip(']').replace(' ', '').split(',') for lkid in
                lkid_list_test]
    
            lkid_list = lkid_list + lkid_list_test
            del lkid_list_test
    
            print(lkid_list[0])
    
            model = models.Word2Vec(lkid_list, workers=8, min_count=0, window=2, size=w2v_size)
            model.save('model_w2v_feanum_' + str(train_rnn_listnum))  #
            model = models.Word2Vec.load('model_w2v_feanum_' + str(train_rnn_listnum))  #
            # print(model.wv.vocab)
            print("finsh! w2v!" + str(train_rnn_listnum))
            print("finsh! w2v!" + str(train_rnn_listnum))
            print("finsh! w2v!" + str(train_rnn_listnum))
            lkid_vecs = []
            for lkid in tqdm(lkid_list):
                lkid_vecs.append(to_text_vector(lkid, model=model))
            del lkid_list
            gc.collect()
            lkid_vecs = np.array(lkid_vecs)
            lkid_vecs = pd.DataFrame(lkid_vecs)
            lkid_vecs.columns = ["w2v_fea_" + str(train_rnn_listnum) + '_' + str(x) for x in range(w2v_size)]
    
            lkid_vecs.to_pickle(
                r'./giscup_2021/giscup_2021/lkid_vecs_fea_' + str(train_rnn_listnum) + '.pkl')
            if is_Test:
                lkid_vecs = lkid_vecs.head(148457)
            data = pd.concat([lkid_vecs, data], axis=1)
            del lkid_vecs
            gc.collect()
        else:
            lkid_vecs = pd.read_pickle(r'./giscup_2021/giscup_2021/lkid_vecs_fea_' + str(
                train_rnn_listnum) + '.pkl')  # , nrows = 148457
            if is_Test:
                lkid_vecs = lkid_vecs.head(148457)
            data = pd.concat([lkid_vecs, data], axis=1)
            del lkid_vecs
            gc.collect()
    else:
        pass
    return data


data = w2v_fea(data, train_rnn_listnum = 0, w2v_size = 32,is_w2v=is_w2v, is_save_w2v=is_save_w2v)
#data = w2v_fea(data, train_rnn_listnum = 3, w2v_size = 16,is_w2v=is_w2v, is_save_w2v=is_save_w2v)


from sklearn.decomposition import PCA
def pca_rnn(data,train_rnn_listnum=0, is_pca_rnn = False,pca_size= 8):
    if is_pca_rnn:
        pca = PCA(n_components=pca_size)

        X = np.concatenate([train_rnn[:, :, train_rnn_listnum], df_test_rnn[:, :, train_rnn_listnum]], axis=0)
        X = pca.fit_transform(X) 
        pca_vec = pd.DataFrame(X)
        pca_vec.columns = ["pca_fea_" + str(train_rnn_listnum) + '_' + str(x) for x in range(pca_size)]
        data = pd.concat([pca_vec, data], axis=1)
        del pca_vec, X
        gc.collect()
        print("pca.explained_variance_ratio_:",pca.explained_variance_ratio_)
    return data
    
#data = pca_rnn(data, train_rnn_listnum = 3, is_pca_rnn = True)
#data = pca_rnn(data, train_rnn_listnum = 1, is_pca_rnn = True)
#data = pca_rnn(data, train_rnn_listnum = 2, is_pca_rnn = True)

if is_w2v_cross:
      lkid_vecs = pd.read_pickle(r'./giscup_2021/all_day_ctoid_vecs.pkl')  # , nrows = 148457
      if is_Test:
          lkid_vecs = lkid_vecs.head(148457)
      data = pd.concat([lkid_vecs, data], axis=1)
      print("cross", lkid_vecs.shape)
      del lkid_vecs
      gc.collect()

x_graph_fea = ["ratio_"+str(x) for x in range(64)]#+["status_"+str(x) for x in range(64)]
useless_fea = ['order_id','ata', 'date', 'd_s', 'cross_num_ata_mean']+x_graph_fea
data, sparse_features, dense_features = recognize_feature(data, categorical_columns, lable=useless_fea)




fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                       for feat in sparse_features] + [DenseFeat(feat, 1,)
                      for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
data = data[list(set(data.columns) - set(useless_fea))+label]
df_train_head = data[0:train_index]
df_test_head = data[train_index:]
df_test_head.reset_index(drop = True,inplace = True)
del data
gc.collect()

################################
#'scale feat'
scaled_features=[]
for col in tqdm(df_train_head.columns):
    if col in dense_features:
        scaled_features.append(col)
len(scaled_features)
means=np.mean(df_train_head[scaled_features].values,axis=0)
stds=np.std(df_train_head[scaled_features].values,axis=0)

for i,col in tqdm(enumerate(scaled_features)):
    df_train_head.loc[:,col]=(df_train_head.loc[:,col]-means[i])/stds[i]
    df_test_head.loc[:,col]=(df_test_head.loc[:,col]-means[i])/stds[i]
del means, stds, scaled_features
gc.collect()
################################


folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)
oof = np.zeros(len(df_train_head))

features = [f for f in df_test_head.columns if f not in useless_fea]

predictions = np.zeros(len(df_test_head))
start = time.time()
feature_importance_df = pd.DataFrame()
start_time = time.time()
score = [0 for _ in range(folds.n_splits)]

# model = WDR(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=hidden_layer, dnn_use_bn=False, task='regression')
print(dense_features)
print(sparse_features)



import math
class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, df_train_head, train_rnn, train_rnn_crosses, batch_size=n_batch, shuffle=True, is_train = True):
        self.batch_size = batch_size
        self.datas = df_train_head
        self.train_rnn = train_rnn
        self.train_rnn_crosses = train_rnn_crosses
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle
        self.is_train = is_train

    def __len__(self):
        return math.ceil(len(self.datas) / float(self.batch_size))
        
    def __getitem__(self, index):
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = self.datas.iloc[batch_indexs]
        trn = {name:batch_data[name].values for name in feature_names}
        trn['hist_link_id'] = self.train_rnn[batch_indexs]
        trn['hist_cross_id'] = self.train_rnn_crosses[batch_indexs]
        if self.is_train:
            return trn, batch_data[label].values
        else:
            return trn

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train_head.values)):
    print("fold n {}".format(fold_))
    with strategy.scope():
        model = WDR(linear_feature_columns, dnn_feature_columns, is_rnn = is_RNN, rnn_shape=(links_size, train_rnn.shape[2]), rnn_crosses_shape=(crosses_size, train_rnn_crosses.shape[2]), rnn_dmodel=rnn_dmodel, emb_add = emb_add, link_emb_size= link_emb_size, cross_emb_size = cross_emb_size, dnn_hidden_units=hidden_layer, cin_layer_size = conv_layer, dnn_use_bn=False, task='regression')
        
    s_train=DataGenerator(df_train_head.iloc[trn_idx], train_rnn[trn_idx,:,:], train_rnn_crosses[trn_idx,:,:], n_batch)
    s_val=DataGenerator(df_train_head.iloc[val_idx], train_rnn[val_idx,:,:], train_rnn_crosses[val_idx,:,:],n_batch)

    #trn_data = df_train_head.iloc[trn_idx][features]
    #trn_rnn = train_rnn[trn_idx,:,:]
    #trn_rnn_crosses = train_rnn_crosses[trn_idx,:,:]
    #trn_y = df_train_head.iloc[trn_idx][label].values
    #trn = {name:trn_data[name].values for name in feature_names}
    #del trn_data
    #gc.collect()
    #trn['hist_link_id'] = trn_rnn
    #del trn_rnn
    #gc.collect()
    #trn['hist_cross_id'] = trn_rnn_crosses
    #del trn_rnn_crosses
    #gc.collect()

    print("------flod:",fold_," ---starting the training---")
        
    learning_rate = CustomSchedule(hidden_layer[1])
    print(learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    reduce_lr = ReduceLROnPlateau(monitor='val_mape', factor=0.1, patience=2, min_lr=min_LR, mode='min')
    es = tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_mape', mode='min',
              restore_best_weights=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),#optimizer,
        loss='mape',
        metrics=['mape']
    )
    #if os.path.isfile("./savemodel/wdr_model_fold_"+str(fold_)+".h5"):
    #     model.load_weights("./savemodel/wdr_model_fold_"+str(fold_)+".h5")
    #else:
    model.fit(s_train, #trn, trn_y,
              validation_data=(s_val),#(val, val_y),
              epochs=n_epoch,
              batch_size=n_batch,
              #workers=4,
              shuffle=True,
              #use_multiprocessing=True,
              callbacks=[reduce_lr, es],verbose=1)
    model.save_weights("./savemodel/wdr_model_fold_"+str(fold_)+".h5")
    del s_train, s_val, trn_idx
    gc.collect()
    
    
    #val_data = df_train_head.iloc[val_idx][features]
    #val_rnn = train_rnn[val_idx,:,:]
    #val_rnn_crosses = train_rnn_crosses[val_idx,:,:]
    #val_y = df_train_head.iloc[val_idx][label].values
    #val = {name:val_data[name].values for name in feature_names}
    #del val_data
    #gc.collect()
    #val['hist_link_id'] = val_rnn
    #del val_rnn
    #gc.collect()
    #val['hist_cross_id'] = val_rnn_crosses
    #del val_rnn_crosses
    #gc.collect()
    s_val=DataGenerator(df_train_head.iloc[val_idx], train_rnn[val_idx,:,:],
                      train_rnn_crosses[val_idx,:,:],n_batch, is_train = False)

    oof[val_idx] = np.squeeze(model.predict(s_val))

    # we perform predictions by chunks
    initial_idx = 0
    chunk_size = 1000000
    current_pred = np.zeros(len(df_test_head))
    while initial_idx < df_test_head.shape[0]:
        if is_Test:
            break
        final_idx = min(initial_idx + chunk_size, df_test_head.shape[0])
        idx = range(initial_idx, final_idx)
        test_data = df_test_head.iloc[idx][features]
        tes = {name: test_data[name].values for name in feature_names}
        tes['hist_link_id'] = df_test_rnn[idx,:,:]
        tes['hist_cross_id'] = df_test_rnn_crosses[idx,:,:]
        #s_test=DataGenerator(df_test_head.iloc[idx][features], df_test_rnn[idx,:,:], df_test_rnn_crosses[idx,:,:],n_batch, is_train = False)
        
        current_pred[idx] = np.squeeze(model.predict(tes, verbose=1))
        initial_idx = final_idx
    print(current_pred[:5])
    tf.keras.backend.clear_session()
    predictions += current_pred / min(folds.n_splits, max_iter)
    if is_Test:
        del current_pred, 
    else:
        del current_pred, tes
    gc.collect()
    print("time elapsed: {:<5.2}s".format((time.time() - start_time) / 3600))
    score[fold_] = mape(np.squeeze(df_train_head.iloc[val_idx][label].T.values), oof[val_idx])
    print("*"*10,"flod:",fold_," mape:", score[fold_],"*"*10)
    if fold_ == max_iter - 1: break

if (folds.n_splits == max_iter):
    print("CV score: {:<8.5f}".format(mape(np.squeeze(df_train_head[label].T.values), oof)))
else:
    print("CV score: {:<8.5f}".format(sum(score) / max_iter))

submission['result'] = predictions
submission.to_csv('./subs/simple_wdr_lk_sub.csv',index=False)
