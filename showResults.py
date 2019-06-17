#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Explanations will be provided later.
# Escrito por Agustin Urquiza
# Contacto: agustin.h.urquiza@gmail.com
# --------------------------------------------------------

import random
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from SegmentationSelectiveSearch.selective_search import selective_search

from model import ModelBase
from auxiliares import area, procesar, iou, save, predictBox


def parser():
    """ Funcion encargada de solicitar los argumentos de entrada.

        Returns:
            <class 'argparse.Namespace'>: Argumentos ingresados por el usuario.
    """

    # Argumentos de entrada permitidos.
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--fword', type=str, required=True,
                        help="""Archivo donde se encuentra word2vec.""")

    parser.add_argument('-u', '--funseen', type=str, required=True,
                        help="""Archivo donde se encuetran las clases no vistas.""")

    parser.add_argument('-t', '--dtest', type=str, required=True,
                        help="""Directorio de las imagenes de test.""")

    parser.add_argument('-fb', '--fboxt', type=str, required=True,
                        help="""Arvhivo donde estan los boundingbox de test.""")

    parser.add_argument('-fm', '--fmodel', type=str, required=True,
                        help="""Arvhivo donde se ecnuentra el modelo pre-entrenado.""")

    parser.add_argument('-s', '--save', type=bool, required=True,
                        help="""True se guarda las imagenes de salida, False solo se muestran.""")

    args = parser.parse_args()

    return args


def drawRectangle(img, boxs_t, boxs_p, unseenName):
    """ Esta funicion dibuja los rectangulos en la imagen

        Args:
            img (np.array): Imagen original.
            boxs_t (List): Bounding box verdaderos.
            boxs_p (List): Bounding box propuestos.
            unseen (List): Clases no vistas.
        Returns:
            img_t, img_p: Imagenes con los rectangulos ya dibujados.
    """
    img_p = img.copy()
    img_t = img.copy()
    for b in boxs_t:
        clas = unseenName[str(b[1])]
        x1, x2, y1, y2 = b[0][0], b[0][2], b[0][1], b[0][3]
        cv2.rectangle(img_t, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_t, clas, (x1+10, y1+10), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                    (0, 255, 0), 1)

    for b in boxs_p:
        clas = unseenName[str(b[1])]
        x1, x2, y1, y2 = b[0][0], b[0][2], b[0][1], b[0][3]
        cv2.rectangle(img_p, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_p, clas, (x1+10, y1+10), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                    (255, 0, 0), 1)

    return img_t, img_p


def main():
    global NCOLS, NFILS, NMSIGNORE
    SCALA = 1 # TUNING maximas propuestas.
    STHSLD = 0.99 # TUNING maximas propuestas.
    IGNORAR = 0.8 # TUNING.
    NMSIGNORE = 0.2 # TUNING.
    NCOLS, NFILS = 299, 299

    args = parser()
    FILEWORD = args.fword
    FILEUNSEEN = args.funseen
    DIRTEST = args.dtest
    FILEBOXT = args.fboxt
    FILEMODEL = args.fmodel
    SAVE = args.save

    boxs = json.load(open(FILEBOXT))
    unseenName = json.load(open(FILEUNSEEN))
    words = json.load(open(FILEWORD))

    unseen = [(k, v) for k, v in words.items() if k in unseenName.keys()]
    nomb = random.choice(glob(DIRTEST + "*")) # Elige una imagen al azar en el directorio.
    nomb = nomb.split('/')[-1]
    img = cv2.imread(DIRTEST + nomb)
    tam = img.shape[0] * img.shape[1]

    # Extrae los bb true para la imagen elegida.
    boxs_t = list(filter(lambda x: x['img_name'] == nomb, boxs))[0]['boxs']
    boxs_t = [(b['box'], b['class']) for b in boxs_t]

    model = ModelBase(compile=False)
    model.load_weights(FILEMODEL)

    resNet = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
    _, R = selective_search(img, colour_space='rgb', scale=SCALA, sim_threshold=STHSLD)
    R = np.array([procesar(r) for r in R if area(r) < (IGNORAR*tam)])

    boxs_p = predictBox(img, R, unseen, model, resNet)
    # Macth indice de clases a numero de clases.
    boxs_p = [(i[0], int(list(unseenName)[i[1]])) for i in boxs_p]

    img_t, img_p = drawRectangle(img, boxs_t, boxs_p, unseenName)

    if SAVE:
        cv2.imwrite('True-' + nomb, img_t)
        cv2.imwrite('Predict-' + nomb, img_p)
    else:
        fig = plt.figure()
        ax1 = plt.subplot(121)
        plt.imshow(img_t)
        ax2 = plt.subplot(122)
        plt.imshow(img_p)
        plt.show()


if __name__ == "__main__":
    main()
