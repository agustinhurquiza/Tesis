#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Explanations will be provided later.
# Escrito por Agustin Urquiza
# Contacto: agustin.h.urquiza@gmail.com
# --------------------------------------------------------

import argparse
from glob import glob
from random import sample
import numpy as np
import json
import itertools
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

    args = parser.parse_args()

    return args


def main():
    args = parser()

    DIRDATA = args.dirdata
    FILEWORD = args.fword
    FILESEEN = args.fseen

    OUTSIZE = 300
    INSIZE = 2048

    seen = json.load(open(FILESEEN)).keys()
    words = json.load(open(FILEWORD))
    words = {k:v for k,v in words.items() if k in seen}

    # Carga la data pre-procesadas.
    X = np.empty((0, INSIZE))
    Y = np.empty((0, OUTSIZE))
    for file in set([item[:-len('-_.mat')] for item in glob(DIRDATA+'/*')]):
        X = np.concatenate((X, load(file + '-X.mat')), axis=0)
        Y = np.concatenate((Y, load(file + '-Y.mat')), axis=0)

    resultado = {}
    for k, v in words.items():
        items = [i for i, y in enumerate(Y) if np.array_equal(y, v)]
        resultado[k] = np.array(sample(X[items].tolist(), 1000))

    print("Termino de preprocesamiento................")
    idem = []
    for k,v in resultado.items():
        r = 0
        for _ in range(1000):
            x1,x2 = sample(list(v), 2)
            r += x1.dot(x2)
        idem.append((k,r/1000.0))

    dist = []
    for (k1, k2) in itertools.combinations(resultado, 2):
        cl1 = resultado[k1]
        cl2 = resultado[k2]
        r = 0
        for _ in range(1000):
            x1 = sample(list(cl1), 1)[0]
            x2 = sample(list(cl2), 1)[0]
            r += x1.dot(x2)
        dist.append(((k1, k2), r/1000.0))
    print(idem)
    print('--------------------------------------')
    print(dist)


if __name__ == "__main__":
    main()
