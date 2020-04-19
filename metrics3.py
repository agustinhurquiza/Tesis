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
from auxiliares import predictBox, procesar, iou, save, extract_boxes_edges


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
    IOUT = 0.5 # TUNING iou para metricas.
    KRECALL = 100 # TUNING cantidad de propuestas que se concidera.
    NCOLS, NFILS = 224, 224
    OUTSIZE = 300
    INSIZE = 512
    MODEL_EDGE = 'bin/bing-model.yml.gz'

    maxBoxes = 1000
    minScore = 0.07
    sminBoxArea = [1, 100, 1000, 10000]
    smaxAspectRatio = [2, 3, 4]
    sedgeMinMag = [1e-08, 1e-07, 1e-06, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01]
    sedgeMergeThr = [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 2e-01, 3e-01, 4e-01, 5e-01]
    sclusterMinMag = [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 2e-01, 3e-01, 4e-01, 5e-01]
    salpha = [i for i in np.arange(0, 1.06, 0.5)]
    sbeta = [i for i in np.arange(0, 1.06, 0.05)]
    seta = [i for i in np.arange(0, 1.06, 0.05)]
    skappa = [i for i in np.arange(0, 4.5, 0.5)]
    sgamma = [i for i in np.arange(0, 4.5, 0.5)]

    args = parser()
    FILEWORD = args.fword
    FILEUNSEEN = args.funseen
    DIRTEST = args.dtest
    FILEBOXT = args.fboxt
    FILEMODEL = args.fmodel

    boxs = json.load(open(FILEBOXT))
    unseenJson = json.load(open(FILEUNSEEN))
    words = json.load(open(FILEWORD))

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(MODEL_EDGE)
    model = ModelBase(compile=False,OUTSIZE=OUTSIZE, INSIZE=INSIZE)
    model.load_weights(FILEMODEL)
    vgg16 = VGG16(include_top=False, weights='imagenet', pooling='max',
                   input_shape=(NCOLS, NFILS, 3))

    unseenNames = []
    unseenKeys  = []
    for k,v in unseenJson.items():
        unseenNames.append(v)
        unseenKeys.append(k)

    unseen = [(k, words[k]) for k in unseenKeys]

    for minBoxArea in sminBoxArea:
      for maxAspectRatio in smaxAspectRatio:
          for edgeMinMag in sedgeMinMag:
            for edgeMergeThr in sedgeMergeThr:
              for clusterMinMag in sclusterMinMag:
                for alpha in salpha:
                  for beta in sbeta:
                    for eta in seta:
                      for kappa in skappa:
                        for gamma in sgamma:
                            print('\n Args:', maxBoxes, minBoxArea, maxAspectRatio,
                                  minScore, edgeMinMag, edgeMergeThr, clusterMinMag,
                                  alpha, beta, eta, kappa, gamma,'\n')

                            per_img = [0] * 10
                            for nimg, img in enumerate(glob(DIRTEST + "*")):
                                nomb = img.split('/')[-1]
                                img = cv2.imread(img)
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                if nimg == 10:
                                    break

                                # Extrae los bb true para la imagen elegida.
                                try:
                                    boxs_t = list(filter(lambda x: x['img_name'] == nomb, boxs))[0]['boxs']
                                except:
                                    continue

                                propuestas, score = extract_boxes_edges(edge_detection, img,
                                                                        maxBoxes, minBoxArea,
                                                                        maxAspectRatio,
                                                                        minScore, edgeMinMag,
                                                                        edgeMergeThr, clusterMinMag,
                                                                        alpha, beta, eta, kappa, gamma)

                                propuestas = [procesar(r) for r in propuestas]
                                propuestas = np.array(propuestas)

                                flgas = [0] * len(boxs_t)
                                for i, b in enumerate(boxs_t):
                                    for bb in propuestas:
                                        if iou(bb, b['box']) >= 0.5:
                                            flgas[i] = flgas[i] + 1

                                print(nomb, flags)
                                if all(flgas) > 0:
                                    per_img[nimg] = True

                            if all(per_img):
                                print('Este modelo puede ser usado')

if __name__ == "__main__":
    main()
