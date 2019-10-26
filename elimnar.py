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

    parser.add_argument('-w', '--fword', type=str, required=True,
                        help="""Archivo donde se encuentran las words embeding.""")

    parser.add_argument('-s', '--fseen', type=str, required=True,
                        help="""Archivo donde se encuentran las clases vistas.""")

    args = parser.parse_args()

    return args


def main():
    args = parser()
    FILEWORD = args.fword
    FILESEEN = args.fseen

    OUTSIZE = 300

    dirs = [('Data/res-32-img-m',  2048), ('Data/vgg-32-img-n',  512), ('Data/res-299-img-a',  2048),
            ('Data/res-150-res-a',  2048), ('Data/res-32-img-n',  2048), ('Data/vgg-224-img-n',  25088),
            ('Data/inc-150-img-m',  1536), ('Data/vgg-32-vgg-n',  512), ('Data/vgg-150-vgg-a',  512),
            ('Data/res-68-img-a',  2048), ('Data/inc-75-img-a',  1536), ('Data/inc-299-img-a',  1536),
            ('Data/res-68-img-m',  2048), ('Data/res-299-img-m',  2048), ('Data/inc-150-img-n',  13824),
            ('Data/res-299-res-m',  2048), ('Data/res-32-img-a',  2048), ('Data/res-299-img-n',  204800),
            ('Data/vgg-224-vgg-a',  512), ('Data/res-32-res-m',  2048), ('Data/res-68-res-n',  18432),
            ('Data/vgg-68-vgg-a',  512), ('Data/vgg-150-vgg-n',  8192), ('Data/inc-299-img-n',  98304),
            ('Data/vgg-224-img-a',  512), ('Data/res-150-img-n',  51200), ('Data/vgg-32-vgg-a',  512),
            ('Data/res-32-res-a',  2048), ('Data/res-68-res-a',  2048), ('Data/res-32-res-n',  2048),
            ('Data/inc-75-img-n',  1536), ('Data/vgg-150-img-n',  8192), ('Data/vgg-32-img-m',  512),
            ('Data/res-299-res-a',  2048), ('Data/vgg-150-img-m',  512), ('Data/inc-150-img-a',  1536),
            ('Data/vgg-224-img-m',  512), ('Data/res-299-res-n',  204800), ('Data/res-150-img-m',  2048),
            ('Data/res-68-res-m',  2048), ('Data/vgg-150-img-a',  512), ('Data/vgg-224-vgg-n',  25088),
            ('Data/res-150-res-n',  51200), ('Data/vgg-68-vgg-n',  2048), ('Data/inc-299-img-m',  1536),
            ('Data/vgg-150-vgg-m',  512), ('Data/inc-75-img-m',  1536), ('Data/vgg-224-vgg-m',  512),
            ('Data/res-150-img-a',  2048), ('Data/res-68-img-n',  18432), ('Data/vgg-68-vgg-m',  512),
            ('Data/vgg-68-img-n',  2048), ('Data/vgg-32-vgg-m',  512), ('Data/vgg-32-img-a',  512),
            ('Data/vgg-68-img-m',  512), ('Data/vgg-68-img-a',  512), ('Data/res-150-res-m',  2048)]

    seen = json.load(open(FILESEEN)).keys()
    words = json.load(open(FILEWORD))
    words = {k:v for k,v in words.items() if k in seen}

    for DIRDATA, INSIZE in dirs:
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
        print(DIRDATA, '---------------------------------------------------------------------------')
        print(idem)
        print('------------------------------------dist---------------------------------------')
        print(dist)


if __name__ == "__main__":
    main()
