#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Explanations will be provided later.
# Escrito por Agustin Urquiza
# Contacto: agustin.h.urquiza@gmail.com
# --------------------------------------------------------

from keras.layers import Input, Dense
from keras.models import Model
from keras.metrics import categorical_accuracy as cacc
import keras.backend as K
from keras.optimizers import Adam
from tensorflow import float32


def custom_loss(W_Clases):
    """ Esta funicion genera la loss function del modelo, esta misma depende de
        las clases vistas, es decir depende del data set.

        Args:
            W_Clases (List): Una lista de tensores (tensorflow) de cada clase
                             vista.
        Returns:
            function: Lost function lossf(y_true, y_pred).
    """
    def lossf(y_true, y_pred):
        loss = 0
        Sii = K.sum(y_true * y_pred, axis=-1)

        for w_clase in W_Clases:
            Sij = K.sum(y_pred * w_clase, axis=-1)
            loss += K.maximum(K.cast(0, float32), K.cast(1, float32) - Sii + Sij)

        #loss -= 1.0
        return loss

    return lossf


def ModelBase(W_Clases=[], OUTSIZE=300, INSIZE=1536, compile=True, lr=0.01):
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
    inputs = Input(shape=(INSIZE,))

    x = Dense(OUTSIZE, activation='relu')(inputs)

    model = Model(inputs=inputs, outputs=x)
    if compile:
        model.compile(optimizer=adam, loss=custom_loss(W_Clases), metrics=[cacc])
    model.summary()
    return model
