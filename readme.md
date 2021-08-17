https://sigspatial2021.sigspatial.org/sigspatial-cup/

1. DIDI_keras_code_1222为keras版本的WDR模型调参后线上得分是0.122053374155646与0.122285919839553，分别存入./subs中
2. DIDI_lgb_code_1379为lightGBM模型，线上得分为0.137901198225055
3. pred_2021_07_24_09_40为pytorch版的类似于mlp+lstm模型，线上0.125209095406921

三者融合merage.py最终线上得分0.121501172396437，b榜0.12177，排名5/1173 