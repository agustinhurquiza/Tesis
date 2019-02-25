#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Explanations will be provided later.
# Escrito por Agustin Urquiza
# Contacto: agustin.h.urquiza@gmail.com
# --------------------------------------------------------

from glob import glob
import argparse
import json
import cv2
import tensorflow as tf
import numpy as np
from random import choices, randint
from sklearn.preprocessing import normalize
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from SegmentationSelectiveSearch.selective_search import selective_search


def process(r):
    """ Funcion formatear los bounding box en [x1, y1, x2, y2].
        Args:
            r (dict): Bounding box sin formatear.
        Returns:
            [int]: Las cuatro cordenadas del bounding box.
    """
    x1 = r['x_min']
    y1 = r['y_min']
    x2 = r['width'] + x1
    y2 = r['height'] + y1

    return [x1, y1, x2, y2]


def area(r):
    """ Funcion encargada de calcular el area de un bounding box.
        Args:
            r (dict): Bounding box.
        Returns:
            float: Area del bounding box.
    """
    x1 = r['x_min']
    y1 = r['y_min']
    x2 = r['width'] + x1
    y2 = r['height'] + y1

    return (x2-x1)*(y2-y1)


def boxsByName(boxs, name):
    """ Devuelve una lista de boundig box asociado a una imagen.
        Args:
            boxs (dict): Diccionario donde se encuentra los boundig boxs
                         de cada imagen.
            name (str): Nombre de la imagen.
        Returns:
            [dict]: Conjunto de bounding box asociado a la imagen..
    """
    boxs = list(filter(lambda x: x['img_name'] == name, boxs))[0]['boxs']

    return boxs


def iou(boxA, boxB):
    """ Funcion encargada de calcular Intersection over Union de dos bounding
         boxs.
        Returns:
            <float>: 0 <= iou <= 1.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def appendValue(X, Y, box, cls, img, model, w2vec):
    """ Agrega los features y los labels para un boundig box.
        Args:
            X (list): Lista de features.
            Y (list): Lista de label.
            box (list): Bounding box.
            cls (String): Clase que pertenece el boundig box.
            img (np.array): Imagen completa.
            model (keras.model): Modelo para extraer los features.
            w2vec (dict): Lista Word2vec para cada clase.

    """
    x1 = box[0]
    x2 = box[2]
    y1 = box[1]
    y2 = box[3]

    x = cv2.resize(img[y1:y2, x1:x2], (NCOLS, NROWS)).reshape(1,NCOLS, NROWS, 3)
    x = normalize(model.predict(x).reshape(1, -1), axis=1)[0].tolist()
    X.append(x)

    if cls != '-1':
        y = w2vec[cls]
        Y.append(y)
    else:
        y = np.zeros([300]).tolist()
        Y.append(y)


def parser():
    """ Funcion encargada de solicitar los argumentos de entrada.

        Returns:
            <class 'argparse.Namespace'>: Argumentos ingresados por el usuario.
    """

    # Argumentos de entrada permitidos.
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dir', type=str, required=True,
                        help="""Directorio donde se encuentran las imagenes.""")

    parser.add_argument('-b', '--bbox', type=str, required=True,
                        help="""Archivo donde se encuentran las bounding box.""")

    parser.add_argument('-s', '--dir_salida', type=str, required=True,
                        help="""Directorio de salida.""")

    parser.add_argument('-n', '--name', type=str, required=True,
                        help="""Nombre del data set.""")

    parser.add_argument('-w', '--word', type=str, required=True,
                        help="""Archivo donde se encuentra los word2vector
                                de las clases.""")

    args = parser.parse_args()

    return args


def main():
    args = parser()
    dir = args.dir
    fileB = args.bbox
    fileW = args.word
    dirO = args.dir_salida
    nameData = args.name

    global NCOLS, NROWS, FEATURE_SIZE

    NCOLS = 299
    NROWS = 299
    FEATURE_SIZE = 1536
    IOU = 0.5
    BACKGROUND = 0.2
    IGNORE = 0.8
    SCALE = 1
    STHSLD = 0.99

    X = []
    Y = []

    images = [{f.split('/')[-1]: cv2.imread(f)} for f in glob(dir + '/*.jpg')]
    boundboxsTruth = json.load(open(fileB, 'r'))
    w2vec = json.load(open(fileW))
    model = InceptionResNetV2(include_top=False, weights='imagenet',
                              pooling='avg')

    for img in images:
        name = list(img.keys())[0]
        img = list(img.values())[0]
        size = img.shape[0] * img.shape[1]
        boxs = boxsByName(boundboxsTruth, name)

        _, R = selective_search(img, colour_space='rgb', scale=SCALE,
                                sim_threshold=STHSLD)

        proposals = [process(r) for r in R if area(r) < (IGNORE*size)]

        for bb in boxs:
            appendValue(X, Y, bb['box'], str(bb['class']), img, model, w2vec)

        for bb in proposals:
            ious = [(i, iou(bb, b['box'])) for i, b in enumerate(boxs)]
            ious.sort(key=lambda x: x[1], reverse=True)
            
            if ious[0][1] > IOU:
                cls =  str(boxs[ious[0][0]]['class'])
                appendValue(X, Y, bb, cls, img, model, w2vec)

            elif ious[0][1] < BACKGROUND:
                appendValue(X, Y, bb, '-1', img, model, w2vec)

            elif ious[0][1] == 0 and randint(0,3):
                appendValue(X, Y, bb, '-1', img, model, w2vec)

    f = open(dirO + nameData + '-feature.json', 'w')
    f.write(json.dumps(X))
    f.close()

    f = open(dirO + nameData + '-label.json', 'w')
    f.write(json.dumps(Y))
    f.close()


if __name__ == "__main__":
    main()
