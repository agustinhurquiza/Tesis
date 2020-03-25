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

import json
import argparse
import cv2
import numpy as np
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from glob import glob
from keras.applications.vgg16 import VGG16
from model import ModelBase
from auxiliares import area, predictBox, procesar, iou, save, extract_boxes_edges


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

    args = parser.parse_args()

    return args


def main():
    IGNORAR = 0.8 # TUNING Elimina las boundingboxes de gran tamaño.
    IOUT = 0.5 # TUNING iou para metricas.
    KRECALL = 100 # TUNING cantidad de propuestas que se concidera.
    NCOLS, NFILS = 224, 224
    OUTSIZE = 300
    INSIZE = 512
    MODEL_EDGE = 'bin/bing-model.yml.gz'
    MAX_BOXS = 600

    args = parser()
    FILEWORD = args.fword
    FILEUNSEEN = args.funseen
    DIRTEST = args.dtest
    FILEBOXT = args.fboxt
    FILEMODEL = args.fmodel

    boxs = json.load(open(FILEBOXT))
    unseenName = json.load(open(FILEUNSEEN))
    words = json.load(open(FILEWORD))

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(MODEL_EDGE)
    model = ModelBase(compile=False,OUTSIZE=OUTSIZE, INSIZE=INSIZE)
    model.load_weights(FILEMODEL)
    vgg16 = VGG16(include_top=False, weights='imagenet', pooling='max',
                   input_shape=(NCOLS, NFILS, 3))

    unseen = [(k, v) for k, v in words.items() if k in unseenName.keys()]

    allBoundingBoxes = BoundingBoxes()
    BoundingBoxesK = BoundingBoxes()

    allgrundtruths = 0

    for nimg, img in enumerate(glob(DIRTEST + "*")):
        print("Imagnes procesadas: ", nimg)
        nomb = img.split('/')[-1]
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tam = img.shape[0] * img.shape[1]

        # Extrae los bb true para la imagen elegida.
        try:
            boxs_t = list(filter(lambda x: x['img_name'] == nomb, boxs))[0]['boxs']
        except:
            print("continue")
            continue

        for (k, b) in enumerate(boxs_t):
            allgrundtruths += 1
            bb = BoundingBox(nomb, b['class'], b['box'][0], b['box'][1], b['box'][2],
                             b['box'][3], CoordinatesType.Absolute, (NCOLS, NFILS),
                             BBType.GroundTruth, format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
            BoundingBoxesK.addBoundingBox(bb)

        propuestas, score = extract_boxes_edges(edge_detection, img, MAX_BOXS)
        indexs = [i for i, s in enumerate(score) if s > 0.07]
        propuestas = propuestas[indexs]
        propuestas = [procesar(r) for r in propuestas]
        #propuestas = [r for r in propuestas if area(r) < (IGNORAR*tam)]
        propuestas = np.array(propuestas)
        
        boxs_p = predictBox(img, propuestas, unseen, model, vgg16, NCOLS=NCOLS, NFILS=NFILS)
        for k, b in enumerate(boxs_p):
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
    tp = 0
    for mc in metricsPerClass:
        tp += mc['total TP']

    print("Recall@%d "% KRECALL)
    print("%.4f" %(tp/allgrundtruths))
if __name__ == "__main__":
    main()
