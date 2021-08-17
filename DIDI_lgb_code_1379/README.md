
## **1. 环境依赖**
- lightgbm>=3.2.1
- gensim>=3.8.3
- python3

## **2. 目录结构**

```
./
├── README.md
├── lgb.py
├── process_data
│   ├── cache
│   ├── w2v_temp_data
|   ├── prepare.sh
|   ├── rule_demo2.py
|   ├── generate_training_data.py
|   ├── w2v_cross.py
|   ├── w2v_link.py
|   ├── w2v_stage.py
├── data
├── subs
```

## **3. 运行流程**
- 进入目录：cd DIDI_code2/process_data
- 数据准备：sh prepare.sh
- 模型训练并预测：执行lgb.py文件

## **4. 模型及特征**
- 模型：lgb
- 参数：
    - learning_rate: 0.05
    - num_leaves: 512-1



   



