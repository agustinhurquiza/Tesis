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
from keras.applications.vgg16 import VGG16

from model import ModelBase
from auxiliares import area, procesar, iou, save, predictBox, extract_boxes_edges, drawRectangle


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

    parser.add_argument('-s', '--save', type=int, required=True,
                        help="""True se guarda las imagenes de salida, False solo se muestran.""")

    args = parser.parse_args()

    return args


def main():
    SCALA = 1 # TUNING maximas propuestas.
    STHSLD = 0.99 # TUNING maximas propuestas.
    IGNORAR = 0.85 # TUNING.

    args = parser()
    FILEWORD = args.fword
    FILEUNSEEN = args.funseen
    DIRTEST = args.dtest
    FILEBOXT = args.fboxt
    FILEMODEL = args.fmodel
    MODEL_EDGE = 'bin/bing-model.yml.gz'
    INSIZE = 512
    NCOLS, NFILS = 299, 299
    SAVE = args.save
    MAX_BOXS = 100

    boxs = json.load(open(FILEBOXT))
    unseenName = json.load(open(FILEUNSEEN))
    words = json.load(open(FILEWORD))

    while(True):
        unseen = [(k, v) for k, v in words.items() if k in unseenName.keys()]
        nomb = random.choice(glob(DIRTEST + "*")) # Elige una imagen al azar en el directorio.
        nomb = nomb.split('/')[-1]
        img = cv2.imread(DIRTEST + nomb)
        tam = img.shape[0] * img.shape[1]

        # Extrae los bb true para la imagen elegida.
        try:
            boxs_t = list(filter(lambda x: x['img_name'] == nomb, boxs))[0]['boxs']
        except:
            continue
        if random.randint(0,20) != 1:
            continue
        boxs_t = [(b['box'], b['class']) for b in boxs_t]
        break

    model = ModelBase(compile=False, INSIZE=INSIZE)
    model.load_weights(FILEMODEL)

    vgg16 = VGG16(include_top=False, weights='imagenet', pooling='max',
                  input_shape=(NCOLS, NFILS, 3))
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(MODEL_EDGE)

    propuestas, score = extract_boxes_edges(edge_detection, img, MAX_BOXS)
    propuestas = [procesar(r) for r in propuestas]
    boxs_p = predictBox(img, propuestas, unseen, model, vgg16)

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
