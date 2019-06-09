#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Explanations will be provided later.
# Escrito por Agustin Urquiza
# Contacto: agustin.h.urquiza@gmail.com
# --------------------------------------------------------

import argparse
from glob import glob
import numpy as np
import json
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import cosine_proximity
from keras.metrics import categorical_accuracy as cacc
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import keras.backend as K
from tensorflow import convert_to_tensor as arrayToTensor
from tensorflow import float32
from auxiliares import save, load


def parser():
    """ Funcion encargada de solicitar los argumentos de entrada.

        Returns:
            <class 'argparse.Namespace'>: Argumentos ingresados por el usuario.
    """

    # Argumentos de entrada permitidos.
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dirdata', type=str, required=True,
                        help="""Directorio donde se encuentan los features.""")

    parser.add_argument('-w', '--fword', type=str, required=True,
                        help="""Archivo donde se encuentran las words embeding.""")

    parser.add_argument('-s', '--fseen', type=str, required=True,
                        help="""Archivo donde se encuentran las clases vistas.""")

    parser.add_argument('-f', '--fileo', type=str, required=True,
                        help="""Archivo de salida del modelo.""")

    parser.add_argument('-n', '--nepocs', type=int, required=True,
                        help="""Numeo de epocas que se corre el modelo.""")

    args = parser.parse_args()

    return args


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
        Sii = K.abs(cosine_proximity(y_pred, y_true))

        for w_clase in W_Clases:
            Sij = K.abs(cosine_proximity(y_pred, w_clase))
            loss += K.maximum(K.cast(0, float32), K.cast(1, float32) - Sii + Sij)

        loss -= 1.0
        return loss

    return lossf


def main():
    args = parser()

    DIRDATA = args.dirdata
    FILEWORD = args.fword
    FILESEEN = args.fseen
    FILEO = args.fileo
    NEPOCS = args.nepocs

    seen = json.load(open(FILESEEN))
    words = json.load(open(FILEWORD))

    resNet = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')

    OUTSIZE = 300
    INSIZE = 1536
    X = np.empty((0, INSIZE))
    Y = np.empty((0, OUTSIZE))
    for file in set([item[:-len('-_.mat')] for item in glob(DIRDATA+'/*')]):
        X = np.concatenate((X, load(file + '-X.mat')), axis=0)
        Y = np.concatenate((Y, load(file + '-Y.mat')), axis=0)

    # Filtra los boundingbox backgraund.
    ZEROS = np.zeros([300])
    items = [i for i, y in enumerate(Y) if not np.array_equal(y, ZEROS)]
    X = X[items]
    Y = Y[items]

    W_Clases = [arrayToTensor(v, 'float32') for k, v in words.items() if k in seen.keys()]

    inputs = Input(shape=(INSIZE,))

    x = Dense(OUTSIZE, activation='relu')(inputs)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss=custom_loss(W_Clases), metrics=[cacc])
    model.summary()
    model.fit(X, Y, epochs=NEPOCS)
    model.save(FILEO)

    print("Finished training the model.")


if __name__ == "__main__":
    main()
