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
from random import choices

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

    parser.add_argument('-s', '--arc_salida', type=str, required=True,
                        help="""Archivo de salida.""")

    args = parser.parse_args()

    return args


def main():
    args = parser()
    dir = args.dir
    fileB = args.bbox
    fileO = args.arc_salida

    images = [{f.split('/')[-1]: cv2.imread(f)} for f in glob(dir + '/*.jpg')]
    boundboxsTruth = json.load(open(fileB, 'r'))
    result = []

    for img in images:
        background = []
        newBoxs = []
        name = list(img.keys())[0]
        img = list(img.values())[0]
        size = img.shape[0] * img.shape[1]
        boxs = boxsByName(boundboxsTruth, name)

        _, R = selective_search(img, colour_space='rgb', scale=320,
                                sim_threshold=0.85)

        proposals = [process(r) for r in R if area(r) < (0.6*size)]

        for bb in proposals:
            ious = [iou(bb, b['box']) for b in boxs]

            try:
                item = next(x[0] for x in enumerate(ious) if x[1] > 0.5)
                newBoxs.append({'box': list(map(int, bb)),
                                'class': boxs[item]['class']})
                continue
            except:
                pass

            try:
                item = next(x[0] for x in enumerate(ious) if 0 < x[1] < 0.2)
                newBoxs.append({'box': list(map(int, bb)), 'class': -1})
                continue
            except:
                pass

            background.append({'box': list(map(int, bb)), 'class': -1})

        if len(background) > 0:
            background = choices(background, k=int(0.8 * len(background)))

        result.append({'img_name': name, 'boxs': boxs + newBoxs + background})

    file = open(fileO, 'w')
    file.write(json.dumps(result))
    file.close()


if __name__ == "__main__":
    main()
