simp_nn.py部分：
step 1. rule_demo.py是处理原始轨迹特征的，输出为每天的train.csv+每天的轨迹序列(-1, link_num, feature)
step 2. link_fea是图特征，直接运行run.sh，把id-time或者id-ratio或者id-status的图用doc2vec转换为vec存到csv里面，效果发现只有id-time有用
step 3. features是把每天的train.csv合并成一个csv，并且造一些统计特征，输出到data_train_feature_lkpro_max_max_max.pkl文件里面
%%% step 4. w2v_cross.py是对cross序列做w2v，输出到all_day_ctoid_vecs.pkl中（w2v_link在simp_nn.py里面）
step 4.然后训练simp_nn.py，可能需要deepCTR的库。在此感谢https://github.com/shenweichen/DeepCTR

