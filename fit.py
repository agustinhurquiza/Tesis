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
import os
from tensorflow import convert_to_tensor as arrayToTensor
from auxiliares import load
from model import ModelBase
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


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

    parser.add_argument('-o', '--diro', type=str, required=True,
                        help="""Directorio de salida del modelo.""")

    parser.add_argument('-n', '--nepocs', type=int, required=True,
                        help="""Numeo de epocas que se corre el modelo.""")

    args = parser.parse_args()

    return args


def main():
    args = parser()

    DIRDATA = args.dirdata
    FILEWORD = args.fword
    FILESEEN = args.fseen
    FILEO = args.diro
    NEPOCS = args.nepocs

    OUTSIZE = 300
    INSIZE = 512

    BATCH_SIZE = 64   # Hiper parametro
    LAMBDA = 10e-3    # Hiper parametro
    MAX_MARGIN = 1    # Hiper parametro
    ACTIVATION = None # Hiper parametro

    seen = json.load(open(FILESEEN))
    words = json.load(open(FILEWORD))

    # Carga la data pre-procesadas.
    X = np.empty((0, INSIZE))
    Y = np.empty((0, OUTSIZE))
    ZEROS = np.zeros([OUTSIZE])

    for file in set([item[:-len('-_.mat')] for item in glob(DIRDATA+'/*')]):
        X_t = load(file + '-X.mat')
        Y_t = load(file + '-Y.mat')

        # Filtra los boundingbox backgraund.
        items = [i for i, y in enumerate(Y_t) if not np.array_equal(y, ZEROS)]
        X_t = X_t[items]
        Y_t = Y_t[items]
        X = np.concatenate((X, X_t), axis=0)
        Y = np.concatenate((Y, Y_t), axis=0)

    print("Se termino de cargar features.")

    W_Clases = arrayToTensor(np.array([v for k, v in words.items() if k in seen.keys()]), 'float32')

    legendLoss = []
    legendMetric = []
    pltL = []
    pltM = []

    for lr in range(1, 9):
        print("Arrancando modelo whit lr")
        lr = 10**-lr
        callbacks = [ModelCheckpoint(filepath=os.path.join(FILEO, '{:.1e}'.format(lr) + '-weights.hdf5'),
                                     monitor='loss', verbose=1,  save_best_only=True,
                                     save_weights_only=False, mode='auto')]

        model = ModelBase(W_Clases, lr=lr, OUTSIZE=OUTSIZE, activation=ACTIVATION,
                          INSIZE=INSIZE, lamb=LAMBDA, max_marg=MAX_MARGIN)
        history = model.fit(X, Y, epochs=NEPOCS, callbacks=callbacks, shuffle=True, batch_size=BATCH_SIZE)

        pltL.append(history.history['loss'])
        pltM.append(history.history['categorical_accuracy'])
        legendLoss += ['Loss-' + str(lr)]
        legendMetric += ['CAc' + str(lr)]
        print("Finalizo de entrenar con lr:" + str(lr) + "\n\n\n\n")

    plt.subplot(2, 1, 1)
    plt.title('Loss and Categorical Accuracy')
    for c in pltL:
        plt.plot(c)
    plt.legend(legendLoss)
    plt.ylabel('Value')

    plt.subplot(2, 1, 2)
    for c in pltM:
        plt.plot(c)
    plt.legend(legendMetric)
    plt.ylabel('Value')
    plt.xlabel('Epoch')

    plt.savefig('graph_loss_Acc.png')
    print("Finished training the model.")


if __name__ == "__main__":
    main()
