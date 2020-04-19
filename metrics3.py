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
    IGNORAR = 0.8 # TUNING Elimina las boundingboxes de gran tamaño.
    IOUT = 0.5 # TUNING iou para metricas.
    KRECALL = 100 # TUNING cantidad de propuestas que se concidera.
    NCOLS, NFILS = 224, 224
    OUTSIZE = 300
    INSIZE = 512
    MODEL_EDGE = 'bin/bing-model.yml.gz'
    MAX_BOXS = 1000
    MIN_SCORE = 0.01

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

    allBoundingBoxes = BoundingBoxes()
    BoundingBoxesK = BoundingBoxes()

    numgrundtruths = 0
    numdetections = 0
    fp = 0
    tp = 0
    my_num = 0
    num_img = 0
    for nimg, img in enumerate(glob(DIRTEST + "*")):
        #print("Imagnes procesadas: ", nimg)
        nomb = img.split('/')[-1]
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tam = img.shape[0] * img.shape[1]

        # Extrae los bb true para la imagen elegida.
        try:
            boxs_t = list(filter(lambda x: x['img_name'] == nomb, boxs))[0]['boxs']
        except:
            #print("continue")
            continue
        
        num_img += 1
        
        for (k, b) in enumerate(boxs_t):
            numgrundtruths += 1

        propuestas, score = extract_boxes_edges(edge_detection, img, MAX_BOXS)
        indexs = [i for i, s in enumerate(score) if s > MIN_SCORE]
        propuestas = propuestas[indexs]
        propuestas = [procesar(r) for r in propuestas]
        #propuestas = [r for r in propuestas if area(r) < (IGNORAR*tam)]
        propuestas = np.array(propuestas)
        
        my = 0
        for b in boxs_t:
            for bb in propuestas:
                if iou(bb, b['box']) >= 0.5:
                    my+=1
        #print('----------------------------', nomb)
        print(my, len(boxs_t), len(propuestas))
        #print('----------------------------')

        boxs_p = predictBox(img, propuestas, unseen, model, vgg16, NCOLS=NCOLS, NFILS=NFILS)
        #print('Luego de: ', len(boxs_p))
        
        flags = [[i['class'], 0, 0] for i in boxs_t]
        for k, b in enumerate(boxs_p):
            numdetections += 1

            ious = [iou(b[0], bt['box']) for bt in boxs_t]

            for n, io in enumerate(ious):
                if io < 0.5:
                    my_num += 1
                else:
                    if int(boxs_t[n]['class']) == int(unseenKeys[b[1]]):
                        flags[n][1] = flags[n][1] + 1
                        tp += 1
                    else:
                        flags[n][2] = flags[n][2] + 1
                        fp += 1
        #print(flags)

    print('Numero de imagenes', num_img)
    print('Fp: ', fp)
    print('Tp: ', tp)
    print('MY: ', my_num)
    print('N detections:', numdetections)
    print('N grundtruths: ', numgrundtruths)

if __name__ == "__main__":
    main()
