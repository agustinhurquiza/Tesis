#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Explanations will be provided later.
# Escrito por Agustin Urquiza
# Contacto: agustin.h.urquiza@gmail.com
# --------------------------------------------------------

import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
libPath = os.path.join(currentPath, 'ObjectDetectionMetrics', 'lib')
if libPath not in sys.path:
    sys.path.insert(0, libPath)

import random
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
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


def main():
    SCALA = 1 # TUNING maxima cantidad de propuestas.
    STHSLD = 0.99 # TUNING maxima cantidad de propuestas.
    IGNORAR = 0.9 # TUNING Elimina las boundingboxes de gran tamaño.
    IOUT = 0.5 # TUNING iou para metricas.
    KRECALL = 100 # TUNING cantidad de propuestas que se concidera.
    NCOLS, NFILS = 299, 299

    args = parser()
    FILEWORD = args.fword
    FILEUNSEEN = args.funseen
    DIRTEST = args.dtest
    FILEBOXT = args.fboxt
    FILEMODEL = args.fmodel

    boxs = json.load(open(FILEBOXT))
    unseenName = json.load(open(FILEUNSEEN))
    words = json.load(open(FILEWORD))

    model = ModelBase(compile=False)
    model.load_weights(FILEMODEL)
    resNet = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')

    unseen = [(k, v) for k, v in words.items() if k in unseenName.keys()]

    allBoundingBoxes = BoundingBoxes()
    BoundingBoxesK = BoundingBoxes()

    for nomb in glob(DIRTEST + "*"):

        nomb = nomb.split('/')[-1]
        img = cv2.imread(DIRTEST + nomb)
        tam = img.shape[0] * img.shape[1]
        # Extrae los bb true para la imagen elegida.
        boxs_t = list(filter(lambda x: x['img_name'] == nomb, boxs))[0]['boxs']

        for (k, b) in enumerate(boxs_t):
            bb = BoundingBox(nomb, b['class'], b['box'][0], b['box'][1], b['box'][2],
                             b['box'][3], CoordinatesType.Absolute, (NCOLS, NFILS),
                             BBType.GroundTruth, format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
            BoundingBoxesK.addBoundingBox(bb)

        _, R = selective_search(img, colour_space='rgb', scale=SCALA, sim_threshold=STHSLD)
        R = np.array([procesar(r) for r in R if area(r) < (IGNORAR*tam)])

        boxs_p = predictBox(img, R, unseen, model, resNet)
        for b in boxs_p:
            bb = BoundingBox(nomb, int(list(unseenName)[b[1]]), b[0][0], b[0][1],
                             b[0][2], b[0][3], CoordinatesType.Absolute, (NCOLS, NFILS),
                             BBType.Detected, b[2], format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
            if k < KRECALL:
                BoundingBoxesK.addBoundingBox(bb)

    evaluator = Evaluator()
    metricsPerClass = evaluator.GetPascalVOCMetrics(allBoundingBoxes,
                                                    IOUThreshold=IOUT,
                                                    method=MethodAveragePrecision.EveryPointInterpolation)

    print("Average precision values per class:\n")
    map = 0
    for mc in metricsPerClass:
        c = mc['class']
        average_precision = mc['AP']
        map += average_precision
        print('Class:%s AP:%.4f' % (c, average_precision))
    print("MAP: %.4f" %(map/len(metricsPerClass)))

    metricsPerClass = evaluator.GetPascalVOCMetrics(BoundingBoxesK,
                                                    IOUThreshold=IOUT,
                                                    method=MethodAveragePrecision.EveryPointInterpolation)
    tp, allDetections = 0, 0
    for mc in metricsPerClass:
        tp += mc['total TP']
        allDetections += mc['total positives']

    print("Recall@%d "% KRECALL)
    print("%.4f" %(tp/allDetections))
if __name__ == "__main__":
    main()
