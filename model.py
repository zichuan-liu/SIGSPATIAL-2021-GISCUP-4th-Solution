
from itertools import chain

import tensorflow as tf
from tensorflow.python.keras.layers import Input
import tensorflow.keras.layers as L
from tensorflow.keras import backend as K
from DeepCTR.deepctr.feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from DeepCTR.deepctr.layers.core import PredictionLayer, DNN
from DeepCTR.deepctr.layers.interaction import FM, CIN
from DeepCTR.deepctr.layers.utils import concat_func, add_func, combined_dnn_input

from DeepCTR.deepctr.layers.interaction import BiInteractionPooling


def WDR(linear_feature_columns, dnn_feature_columns, is_rnn = True, rnn_shape = (100,3), rnn_crosses_shape = (40,2), link_emb_size = 639878, cross_emb_size = 44312, rnn_dmodel = 32,emb_add = True, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),cin_layer_size=(128, 128,), cin_split_half=True, cin_activation='relu',
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns,
        add_fea_columns_shape={'hist_link_id': rnn_shape,
                                 'hist_cross_id': rnn_crosses_shape,
                                 })
    print(features.keys())

    inputs_list = list(features.values())
    inputs_rnn = features["hist_link_id"]
    inputs_rnn_cross = features["hist_cross_id"]

    if is_rnn:
        link_id = inputs_rnn[:,:,0]
        link_id = tf.keras.layers.Embedding(link_emb_size+1, rnn_dmodel)(link_id)
        
        cross_id = inputs_rnn_cross[:,:,0]
        cross_id = tf.keras.layers.Embedding(cross_emb_size+1, rnn_dmodel)(cross_id)
        
        if emb_add:
            link_time = inputs_rnn[:,:,1]
            link_time = tf.expand_dims(link_time, axis=-1) 
            link_time=tf.keras.layers.Dense(rnn_dmodel, use_bias=False)(link_time)
            link_ratio = inputs_rnn[:,:,2]
            link_ratio = tf.expand_dims(link_ratio, axis=-1) 
            link_ratio=tf.keras.layers.Dense(rnn_dmodel, use_bias=False)(link_ratio)
            link_current_status = inputs_rnn[:,:,3]
            link_current_status = tf.keras.layers.Embedding(4+1, rnn_dmodel)(link_current_status) 
            
            inputs_rnn = tf.keras.layers.Add()([                        
                link_id,
                link_time,
                link_ratio,
                link_current_status
            ]) 
        
            cross_time = inputs_rnn_cross[:,:,1]
            cross_time = tf.expand_dims(cross_time, axis=-1) 
            cross_time=tf.keras.layers.Dense(rnn_dmodel, use_bias=False)(cross_time)
            inputs_rnn_cross = tf.keras.layers.Add()([                        
                cross_id,
                cross_time,
            ]) 
        
        else:
            inputs_rnn = tf.keras.layers.Concatenate()([link_id, inputs_rnn[:,:,1:]])
            inputs_rnn = tf.keras.layers.Dense(rnn_dmodel)(inputs_rnn)
            
            inputs_rnn_cross = tf.keras.layers.Concatenate()([cross_id, inputs_rnn_cross[:,:,1:]])
            inputs_rnn_cross = tf.keras.layers.Dense(rnn_dmodel)(inputs_rnn_cross)
    
        #encoder = Encoder(num_layers=2, d_model=rnn_dmodel, num_heads=2, dff=64,  maximum_position_encoding = rnn_shape[0], rate = dnn_dropout)
        
        cnn_layer = tf.keras.layers.Conv1D(filters=rnn_dmodel, kernel_size=4, padding='same')
        cnn_rnn = cnn_layer(inputs_rnn)
        #rnn = encoder(inputs_rnn, True, None)  # (batch_size, inp_seq_len, d_model)
        cnn_rnn = L.Attention()([cnn_rnn, cnn_rnn])
        cnn_rnn = K.sum(cnn_rnn , axis=-2 , keepdims=False)
        cnn_rnn = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(cnn_rnn)
################ gru
        cnn_layer2 = tf.keras.layers.Conv1D(filters=rnn_dmodel*2, kernel_size=2, padding='same')
        #inputs_rnn = cnn_layer2(inputs_rnn)
        #lstm = L.LSTM(16, return_sequences=False, activation='relu')(inputs_rnn)
        gru = L.GRU(rnn_dmodel, return_sequences=True, activation='relu')(inputs_rnn)#cnn_lstm
        gru = K.sum(gru , axis=-2 , keepdims=False)

#################gru2            
        gru2 = L.GRU(rnn_dmodel//2, return_sequences=True, activation='relu')(inputs_rnn_cross)#cnn_lstm
        gru2 = K.sum(gru2 , axis=-2 , keepdims=False)
        
        #gru_logit = tf.keras.layers.Concatenate()([gru, gru2])
        gru_logit = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(gru)
#################logit       
        
    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,seed, support_group=True)

    fm_logit = add_func([FM()(concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in fm_group])

    #fm_input = concat_func(list(chain.from_iterable(group_embedding_dict.values())), axis=1)
    #bi_out = BiInteractionPooling()(fm_input)
    #dnn_input = combined_dnn_input([bi_out], dense_value_list)

    dnn_input = combined_dnn_input(list(chain.from_iterable(group_embedding_dict.values())), dense_value_list)
    #dnn_input = tf.keras.layers.BatchNormalization()(dnn_input)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    if is_rnn:
        dnn_output = tf.keras.layers.Concatenate()([dnn_output, gru, gru2])
        dnn_output = tf.keras.layers.Dense(rnn_dmodel)(dnn_output)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

    final_logit = add_func([dnn_logit, fm_logit, linear_logit])#fm_logit,linear_logit
    if is_rnn:
        final_logit = add_func([final_logit, cnn_rnn, gru_logit])#, gru
        
    #if len(cin_layer_size) > 0:
    #    exFM_out = CIN(cin_layer_size, cin_activation,
    #                   cin_split_half, l2_reg_dnn, seed)(fm_input)
    #    exFM_logit = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(exFM_out)
    #    final_logit = add_func([final_logit, exFM_logit])

        
    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
