simp_nn.py部分：
step 1. rule_demo.py是处理原始轨迹特征的，输出为每天的train.csv+每天的轨迹序列(-1, link_num, feature)
step 2. link_fea是图特征，直接运行run.sh，把id-time或者id-ratio或者id-status的图用doc2vec转换为vec存到csv里面，效果发现只有id-time有用
step 3. features是把每天的train.csv合并成一个csv，并且造一些统计特征，输出到data_train_feature_lkpro_max_max_max.pkl文件里面
step 4. w2v_cross.py是对cross序列做w2v，输出到all_day_ctoid_vecs.pkl中（w2v_link在simp_nn.py里面）
step 5. 然后训练simp_nn1.py/simp_nn1.py，两者只是参数/特征不一样，也可以多跑几遍不同参数的取平均，线上在0.1220-0.1228左右。
模型部分主要思路是WDR，rnn部分采用cnn+attation和GRU，embsize=4，可能需要deepCTR的库。在此感谢https://github.com/shenweichen/DeepCTR

