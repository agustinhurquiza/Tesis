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
from tensorflow import convert_to_tensor as arrayToTensor
from auxiliares import load
from model import ModelBase
import matplotlib.pyplot as plt


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


def main():
    args = parser()

    DIRDATA = args.dirdata
    FILEWORD = args.fword
    FILESEEN = args.fseen
    FILEO = args.fileo
    NEPOCS = args.nepocs

    OUTSIZE = 300
    INSIZE = 1536

    seen = json.load(open(FILESEEN))
    words = json.load(open(FILEWORD))

    # Carga la data pre-procesadas.
    X = np.empty((0, INSIZE))
    Y = np.empty((0, OUTSIZE))
    for file in set([item[:-len('-_.mat')] for item in glob(DIRDATA+'/*')]):
        X = np.concatenate((X, load(file + '-X.mat')), axis=0)
        Y = np.concatenate((Y, load(file + '-Y.mat')), axis=0)

    # Filtra los boundingbox backgraund.
    ZEROS = np.zeros([OUTSIZE])
    items = [i for i, y in enumerate(Y) if not np.array_equal(y, ZEROS)]
    X = X[items]
    Y = Y[items]

    W_Clases = [arrayToTensor(v, 'float32') for k, v in words.items() if k in seen.keys()]

    legendLoss = []
    legendMetric = []
    pltL = []
    pltM = []
    for lr in range(1, 8):
        lr = 10**-lr
        model = ModelBase(W_Clases, lr=lr)
        history = model.fit(X, Y, epochs=NEPOCS)
        model.save(FILEO+'-lr-'+str(lr)+'.h5')

        pltL.append(history.history['loss'])
        pltM.append(history.history['categorical_accuracy'])
        legendLoss += ['Loss-'+str(lr)]
        legendMetric += ['CAc'+ str(lr)]
        print("Finalizo de entrenar con lr:" + str(lr) + "\n\n\n\n")

    plt.subplot(2, 1, 1)
    plt.title('Loss and Categorical Accuracy')
    plt.plot(pltL)
    plt.legend(legendLoss)
    plt.ylabel('Value')

    plt.subplot(2, 1, 2)
    plt.plot(pltM)
    plt.legend(legendMetric)
    plt.ylabel('Value')
    plt.xlabel('Epoch')

    plt.savefig('graph_loss_Acc.png')
    print("Finished training the model.")


if __name__ == "__main__":
    main()