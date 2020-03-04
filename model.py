#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Explanations will be provided later.
# Escrito por Agustin Urquiza
# Contacto: agustin.h.urquiza@gmail.com
# --------------------------------------------------------

from keras.layers import Dense, Input
from keras.models import Model
from keras.metrics import categorical_accuracy as cacc
import keras.backend as K
from keras.optimizers import Adam
from tensorflow import float32, int64
from tensorflow import convert_to_tensor as arrayToTensor
from tensorflow.linalg import diag_part as diag
from keras.initializers import RandomUniform
from auxiliares import save

def custom_loss(W_Clases, model, inputt, lamb, max_marg):
    """ Esta funicion genera la loss function del modelo, esta misma depende de
        las clases vistas, es decir depende del data set.

        Args:
            W_Clases (List): Una lista de tensores (tensorflow) de cada clase
                             vista.
        Returns:
            function: Lost function lossf(y_true, y_pred).
    """
    def lossf(y_true, y_pred):
        WT = arrayToTensor(model.get_weights()[0].T, 'float32')
        input_r = K.dot(y_pred, WT)
        input_r = K.l2_normalize(input_r, -1)
        loss_resc = K.batch_dot(input_r, K.transpose(inputt))
        
        y_pred = K.l2_normalize(y_pred, -1)
        
        Sii = K.batch_dot(y_pred, K.transpose(y_true))
        Sij = K.dot(y_pred, K.transpose(W_Clases))
        
        loss = K.sum(K.maximum(K.cast(0, float32), K.tf.add((K.cast(max_marg, float32) - Sii), Sij)), axis=-1)

        return (loss + 1.0) + (lamb * loss_resc)

    return lossf


def ModelBase(W_Clases=[], OUTSIZE=300, INSIZE=1536, activation='relu', compile=True, lr=0.01, lamb=0.1, max_marg=1):
    """ Esta funicion genera el modelo. ModelBase consta de una sola capa full
        connected del tamaño de las features a el tamaño de word2vec.

        Args:
            W_Clases (List): Una lista de tensores (tensorflow) de cada clase vista.
            OUTSIZE (int): Tamaño de salida de la full connected.
            INSIZE (int): Tamaño de entrada de la full connected.
            compile (bool): En True se utiliza si se quiere complir el modelo y
                            en False si solo se quiere tener la estructura del
                            modelo.
        Returns:
            model: Modelo keras.
    """

    adam = Adam(lr=lr)

    inputt = Input(shape=(INSIZE,), name='Input_layer')
    output = Dense(OUTSIZE, input_shape=(INSIZE,), activation=activation, name='Output_layer',
                   kernel_initializer=RandomUniform(minval=-1, maxval=1))(inputt)
    
    model = Model(inputt, output)
    loss = custom_loss(W_Clases, model, inputt, lamb, max_marg)
    if compile:
        model.compile(optimizer=adam, loss=loss, metrics=[cacc])
        model.summary()
    return model

