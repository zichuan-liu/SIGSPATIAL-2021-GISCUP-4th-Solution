step 1. rule_demo.py是处理原始轨迹特征的，输出为每天的train.csv+每天的轨迹序列(-1, link_num, feature)
step 2. link_fea是图特征，把id-time或者id-ratio或者id-status的图用doc2vec转换为vec存到csv里面，效果发现只有id-time有用
step 3. features是把每天的train.csv合并成一个csv，并且造一些统计特征，输出到data_train_feature_lkpro_max_max_max.pkl文件里面
#### step 4. w2v_cross.py是对cross序列做w2v，输出到all_day_ctoid_vecs.pkl中（w2v_link在simp_nn.py里面）
step 4.然后训练simp_nn.py，可能需要deepCTR的库。在此感谢https://github.com/shenweichen/DeepCTR

simp_nn.py模型调参后线上得分是0.122053374155646与0.122285919839553，分别存入./subs中
----------------------
simp_lgb.py线上得分为0.137901198225055
----------------------
pred_2021_07_24_09_40为pytorch版的类似于mlp+lstm模型，线上0.125209095406921

三者融合merage.py最终线上得分0.121501172396437，b榜0.12177，排名第五