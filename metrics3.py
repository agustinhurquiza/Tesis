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
from random import choice
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

    maxBoxes = 10000
    minScore = 0.05
    sminBoxArea = [10000, 1000] 
    smaxAspectRatio = [3, 2, 4]
    sedgeMinMag =    [1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 0.25, 0.5, 0.75]
    sedgeMergeThr =  [1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 0.25, 0.5, 0.75]
    sclusterMinMag = [1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 0.25, 0.5, 0.75]
    salpha = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    sbeta =  [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    seta =   [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    skappa = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    sgamma = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    imagnes =["DataSets/Imagenes/COCO/Test/COCO_val2014_000000556278.jpg",
              "DataSets/Imagenes/COCO/Test/COCO_val2014_000000091564.jpg",
              "DataSets/Imagenes/COCO/Test/COCO_val2014_000000379760.jpg",
              "DataSets/Imagenes/COCO/Test/COCO_val2014_000000148707.jpg",
              "DataSets/Imagenes/COCO/Test/COCO_val2014_000000561681.jpg",
              "DataSets/Imagenes/COCO/Test/COCO_val2014_000000317310.jpg",
              "DataSets/Imagenes/COCO/Test/COCO_val2014_000000221725.jpg"]

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
    for kk in range(50000):
        edgeMinMag = choice(sedgeMinMag)
        edgeMergeThr = choice(sedgeMergeThr)
        clusterMinMag = choice(sclusterMinMag)
        alpha = choice(salpha)
        beta = choice(sbeta)
        eta = choice(seta)
        kappa = choice(skappa)
        gamma = choice(sgamma)
        minBoxArea = choice(sminBoxArea)
        maxAspectRatio = choice(smaxAspectRatio)
     
        print('\n -> Args:', maxBoxes, minBoxArea, maxAspectRatio,
              minScore, edgeMinMag, edgeMergeThr, clusterMinMag,
              alpha, beta, eta, kappa, gamma)


        per_img = [0] * 7
        count_per_imagen = 0
        for nimg, img in enumerate(imagnes):
            nomb = img.split('/')[-1]
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extrae los bb true para la imagen elegida.
            try:
                boxs_t = list(filter(lambda x: x['img_name'] == nomb, boxs))[0]['boxs']
            except:
                continue
            try:
                propuestas, score = extract_boxes_edges(edge_detection, img,
                                                        maxBoxes=maxBoxes, minBoxArea=minBoxArea,
                                                        maxAspectRatio=maxAspectRatio,
                                                        minScore=minScore, edgeMinMag=edgeMinMag,
                                                        edgeMergeThr=edgeMergeThr, clusterMinMag=clusterMinMag,
                                                        alpha=alpha, beta=beta, eta=eta, kappa=kappa, gamma=gamma)
            
            except:
                propuestas = []
                score = []
                print("Time out Modelo tarda mucho")
                break

            propuestas = [procesar(r) for r in propuestas]
            propuestas = np.array(propuestas)
            #print("Cantidad de prop; ", len(propuestas))
            flgs = [0] * len(boxs_t)
            for i, b in enumerate(boxs_t):
                for bb in propuestas:
                    if iou(bb, b['box']) >= 0.5:
                       flgs[i] = flgs[i] + 1
            
            count_per_imagen = sum([1 for i in flgs if i > 0])
            if all(flgs) > 0:
                per_img[nimg] = True

        if all(per_img):
            print('Este modelo SI puede ser usado: ', count_per_imagen)
            return
        elif count_per_imagen > 20:
            print('Este modelo CASI puede ser usado: ', count_per_imagen)
        else:
            print('Este modelo NO puede ser usado: ', count_per_imagen)

if __name__ == "__main__":
    main()
