import pandas as pd
import numpy as np
from gensim import models
from tqdm import tqdm
import os
def to_text_vector(words, model):
    # words = txt.split(',')
    array = np.asarray([model.wv[w] for w in words if w in words], dtype='float32')
    return array.mean(axis=0)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.decomposition import TruncatedSVD

train_path = r"./giscup_2021/train_fea/"
filenames = os.listdir(train_path)
filenames.sort(key=lambda x: int(x[6:8]))
train_data = []


# test_link_data = np.load('../data/train_fea/20200801_links.npz')
# train_rnn1 = test_link_data['data']

# test_link_data2 = np.load('../data/train_fea/20200802_links.npz')
# train_rnn2 = test_link_data2['data']

# test_link_data = np.load('../data/test_fea/test_cross.npz')
# df_test_rnn = test_link_data['data']
#
# test_link_data = np.load('../data/train_fea/20200801_crosss.npz')
# train_rnn1 = test_link_data['data']
#
# test_link_data2 = np.load('../data/train_fea/20200802_crosss.npz')
# train_rnn2 = test_link_data2['data']

'''
link id
link time
link ratio
link current status


cross id
cross time
'''
#filenames = ["20200801_crosss.npz","20200802_crosss.npz"]
is_w2v = 1
is_save_w2v = 0
if is_w2v:

    if not is_save_w2v:
        lkid_list = []
        for file in tqdm(filenames):
            try:
                if file[9:14] == 'cross':
                    print(file)
                    print(file[9:14])
                    train_link_data = np.load(train_path + file)
                    train_rnn1 = train_link_data['data']
                    lkid_list1 = train_rnn1[:, :, 0].astype(int).tolist()  # his id
                    lkid_list1 = [str(list(filter(lambda x: x != 0, lkid))).strip('[').strip(']').replace(' ', '').split(',')
                                 for
                                 lkid in lkid_list1]
                    lkid_list = lkid_list+lkid_list1
                else:
                    pass
            except:
                pass

        # lkid_list = train_rnn1[:, :, 3].astype(int).tolist()  # his id
        # lkid_list = [str(list(filter(lambda x: x != 0, lkid))).strip('[').strip(']').replace(' ', '').split(',') for
        #              lkid in lkid_list]
        #
        # lkid_list2 = train_rnn2[:, :, 3].astype(int).tolist()  # his id
        # lkid_list2 = [str(list(filter(lambda x: x != 0, lkid))).strip('[').strip(']').replace(' ', '').split(',') for
        #              lkid in lkid_list2]
        # lkid_list = lkid_list+lkid_list2

        test_link_data = np.load('./giscup_2021/test_fea/test_crosses.npz')
        df_test_rnn = test_link_data['data']
        lkid_list_test = df_test_rnn[:, :, 0].astype(int).tolist()  # his id
        lkid_list_test = [str(list(filter(lambda x: x != 0, lkid))).strip('[').strip(']').replace(' ', '').split(',')
                          for lkid in lkid_list_test]

        lkid_list = lkid_list + lkid_list_test
        del lkid_list_test
        print(len(lkid_list))

        print(lkid_list[0])

        model = models.Word2Vec(lkid_list, workers=8, min_count=0, window=2, size=32)
        model.save('model_all_day_cross_w2v')  #
        model = models.Word2Vec.load('model_all_day_cross_w2v')  #

        lkid_vecs = []
        for lkid in tqdm(lkid_list):
            lkid_vecs.append(to_text_vector(lkid, model=model))
        del lkid_list
        gc.collect()
        lkid_vecs = np.array(lkid_vecs)
        lkid_vecs = pd.DataFrame(lkid_vecs)
        lkid_vecs.columns = ["w2v_all_day_cross_" + str(x) for x in range(32)]

        lkid_vecs.to_pickle(r'./giscup_2021/all_day_ctoid_vecs.pkl')
        data = pd.concat([lkid_vecs, data], axis=1)
        del lkid_vecs
        gc.collect()
    else:
        lkid_vecs = pd.read_pickle(r'./giscup_2021/all_day_ctoid_vecs.pkl')  # , nrows = 148457
        #data = pd.concat([lkid_vecs, data], axis=1) #
        del lkid_vecs
        #gc.collect()

else:
    pass