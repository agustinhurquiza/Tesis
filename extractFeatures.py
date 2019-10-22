    #!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Explanations will be provided later.
# Escrito por Agustin Urquiza
# Contacto: agustin.h.urquiza@gmail.com
# --------------------------------------------------------

from glob import glob
import os
import argparse
import json
import cv2
import math
import numpy as np
from random import randint
from sklearn.preprocessing import normalize
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50


from auxiliares import save, load, iou, area, procesar
from SegmentationSelectiveSearch.selective_search import selective_search


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

    try:
        x = cv2.resize(img[y1:y2, x1:x2], (NCOLS, NFILS)).reshape(1, NCOLS, NFILS, 3)
    except:
        return

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
    nombreData = args.name

    vggweights = '../Weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    resweights = '../Weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    global NCOLS, NFILS

    # Tamaño de las imagnes de la entrada del modelo.
    NCOLS = 65
    NFILS = 65
    # Intersecion que debe tener una propuesta con un box.
    IOU = 0.5
    # Intersecion para considerar una propuesta como background.
    BACKGROUND = 0.2
    # Tamaño maximo de las propuetas, en % del tamaño de la imagen.
    IGNORAR = 0.8
    # Parametros de selective search.
    SCALA = 1
    STHSLD = 0.99
    # Probabilidad de backgraund se agrege (1/PROB).
    PROB = 5
    # Maximo de imagenes procesadas sin guardar.
    MAXS = 100

    boundboxs = json.load(open(fileB, 'r'))
    w2vec = json.load(open(fileW))
    modelos = [(VGG16(include_top=False, weights='imagenet', pooling=None , input_shape=(224, 224, 3)), 224, 224, 'vgg-224-img-n'),
               (VGG16(include_top=False, weights='imagenet', pooling=None , input_shape=(150, 150, 3)), 150, 150, 'vgg-150-img-n'),
               (VGG16(include_top=False, weights='imagenet', pooling=None , input_shape=(68, 68, 3)), 68, 68, 'vgg-68-img-n'),
               (VGG16(include_top=False, weights='imagenet', pooling=None , input_shape=(32, 32, 3)), 32, 32, 'vgg-32-img-n'),
               (VGG16(include_top=False, weights='imagenet', pooling='max', input_shape=(224, 224, 3)), 224, 224, 'vgg-224-img-m'),
               (VGG16(include_top=False, weights='imagenet', pooling='max', input_shape=(150, 150, 3)), 150, 150, 'vgg-150-img-m'),
               (VGG16(include_top=False, weights='imagenet', pooling='max', input_shape=(68, 68, 3)), 68, 68, 'vgg-68-img-m'),
               (VGG16(include_top=False, weights='imagenet', pooling='max', input_shape=(32, 32, 3)), 32, 32, 'vgg-32-img-m'),
               (VGG16(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3)), 224, 224, 'vgg-224-img-a'),
               (VGG16(include_top=False, weights='imagenet', pooling='avg', input_shape=(150, 150, 3)), 150, 150, 'vgg-150-img-a'),
               (VGG16(include_top=False, weights='imagenet', pooling='avg', input_shape=(68, 68, 3)), 68, 68, 'vgg-68-img-a'),
               (VGG16(include_top=False, weights='imagenet', pooling='avg', input_shape=(32, 32, 3)), 32, 32, 'vgg-32-img-a'),
               (VGG16(include_top=False, weights=vggweights, pooling=None , input_shape=(224, 224, 3)), 224, 224, 'vgg-224-vgg-n'),
               (VGG16(include_top=False, weights=vggweights, pooling=None , input_shape=(150, 150, 3)), 150, 150, 'vgg-150-vgg-n'),
               (VGG16(include_top=False, weights=vggweights, pooling=None , input_shape=(68, 68, 3)), 68, 68, 'vgg-68-vgg-n'),
               (VGG16(include_top=False, weights=vggweights, pooling=None , input_shape=(32, 32, 3)), 32, 32, 'vgg-32-vgg-n'),
               (VGG16(include_top=False, weights=vggweights, pooling='max', input_shape=(224, 224, 3)), 224, 224, 'vgg-224-vgg-m'),
               (VGG16(include_top=False, weights=vggweights, pooling='max', input_shape=(150, 150, 3)), 150, 150, 'vgg-150-vgg-m'),
               (VGG16(include_top=False, weights=vggweights, pooling='max', input_shape=(68, 68, 3)), 68, 68, 'vgg-68-vgg-m'),
               (VGG16(include_top=False, weights=vggweights, pooling='max', input_shape=(32, 32, 3)), 32, 32, 'vgg-32-vgg-m'),
               (VGG16(include_top=False, weights=vggweights, pooling='avg', input_shape=(224, 224, 3)), 224, 224, 'vgg-224-vgg-a'),
               (VGG16(include_top=False, weights=vggweights, pooling='avg', input_shape=(150, 150, 3)), 150, 150, 'vgg-150-vgg-a'),
               (VGG16(include_top=False, weights=vggweights, pooling='avg', input_shape=(68, 68, 3)), 68, 68, 'vgg-68-vgg-a'),
               (VGG16(include_top=False, weights=vggweights, pooling='avg', input_shape=(32, 32, 3)), 32, 32, 'vgg-32-vgg-a'),

               (ResNet50(include_top=False, weights='imagenet', pooling=None , input_shape=(299, 299, 3)), 299, 299, 'res-299-img-n'),
               (ResNet50(include_top=False, weights='imagenet', pooling=None , input_shape=(150, 150, 3)), 150, 150, 'res-150-img-n'),
               (ResNet50(include_top=False, weights='imagenet', pooling=None , input_shape=(68, 68, 3)), 68, 68, 'res-68-img-n'),
               (ResNet50(include_top=False, weights='imagenet', pooling=None , input_shape=(32, 32, 3)), 32, 32, 'res-32-img-n'),
               (ResNet50(include_top=False, weights='imagenet', pooling='max', input_shape=(299, 299, 3)), 299, 299, 'res-299-img-m'),
               (ResNet50(include_top=False, weights='imagenet', pooling='max', input_shape=(150, 150, 3)), 150, 150, 'res-150-img-m'),
               (ResNet50(include_top=False, weights='imagenet', pooling='max', input_shape=(68, 68, 3)), 68, 68, 'res-68-img-m'),
               (ResNet50(include_top=False, weights='imagenet', pooling='max', input_shape=(32, 32, 3)), 32, 32, 'res-32-img-m'),
               (ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(299, 299, 3)), 299, 299, 'res-299-img-a'),
               (ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(150, 150, 3)), 150, 150, 'res-150-img-a'),
               (ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(68, 68, 3)), 68, 68, 'res-68-img-a'),
               (ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(32, 32, 3)), 32, 32, 'res-32-img-a'),
               (ResNet50(include_top=False, weights=resweights, pooling=None , input_shape=(299, 299, 3)), 299, 299, 'res-299-res-n'),
               (ResNet50(include_top=False, weights=resweights, pooling=None , input_shape=(150, 150, 3)), 150, 150, 'res-150-res-n'),
               (ResNet50(include_top=False, weights=resweights, pooling=None , input_shape=(68, 68, 3)), 68, 68, 'res-68-res-n'),
               (ResNet50(include_top=False, weights=resweights, pooling=None , input_shape=(32, 32, 3)), 32, 32, 'res-32-res-n'),
               (ResNet50(include_top=False, weights=resweights, pooling='max', input_shape=(299, 299, 3)), 299, 299, 'res-299-res-m'),
               (ResNet50(include_top=False, weights=resweights, pooling='max', input_shape=(150, 150, 3)), 150, 150, 'res-150-res-m'),
               (ResNet50(include_top=False, weights=resweights, pooling='max', input_shape=(68, 68, 3)), 68, 68, 'res-68-res-m'),
               (ResNet50(include_top=False, weights=resweights, pooling='max', input_shape=(32, 32, 3)), 32, 32, 'res-32-res-m'),
               (ResNet50(include_top=False, weights=resweights, pooling='avg', input_shape=(299, 299, 3)), 299, 299, 'res-299-res-a'),
               (ResNet50(include_top=False, weights=resweights, pooling='avg', input_shape=(150, 150, 3)), 150, 150, 'res-150-res-a'),
               (ResNet50(include_top=False, weights=resweights, pooling='avg', input_shape=(68, 68, 3)), 68, 68, 'res-68-res-a'),
               (ResNet50(include_top=False, weights=resweights, pooling='avg', input_shape=(32, 32, 3)), 32, 32, 'res-32-res-a'),

               (InceptionResNetV2(include_top=False, weights='imagenet', pooling=None , input_shape=(299, 299, 3)), 299, 299, 'inc-299-img-n'),
               (InceptionResNetV2(include_top=False, weights='imagenet', pooling=None , input_shape=(150, 150, 3)), 150, 150, 'inc-150-img-n'),
               (InceptionResNetV2(include_top=False, weights='imagenet', pooling=None , input_shape=(68, 68, 3)), 68, 68, 'inc-68-img-n'),
               (InceptionResNetV2(include_top=False, weights='imagenet', pooling=None , input_shape=(32, 32, 3)), 32, 32, 'inc-32-img-n'),
               (InceptionResNetV2(include_top=False, weights='imagenet', pooling='max', input_shape=(299, 299, 3)), 299, 299, 'inc-299-img-m'),
               (InceptionResNetV2(include_top=False, weights='imagenet', pooling='max', input_shape=(150, 150, 3)), 150, 150, 'inc-150-img-m'),
               (InceptionResNetV2(include_top=False, weights='imagenet', pooling='max', input_shape=(68, 68, 3)), 68, 68, 'inc-68-img-m'),
               (InceptionResNetV2(include_top=False, weights='imagenet', pooling='max', input_shape=(32, 32, 3)), 32, 32, 'inc-32-img-m'),
               (InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(299, 299, 3)), 299, 299, 'inc-299-img-a'),
               (InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(150, 150, 3)), 150, 150, 'inc-150-img-a'),
               (InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(68, 68, 3)), 68, 68, 'inc-68-img-a'),
               (InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(32, 32, 3)), 32, 32, 'inc-32-img-a')]

    for (model, NCOLS, NFILS, dirs) in models:
        print('start model: ' + dirs)
        dirs = 'Data/' + dirs
        os.mkdir(dirs)
        X = []
        Y = []
        for k, img in enumerate(glob(dir + '/*.jpg')):
            print("Imagenes procesadas: " + str(k+1))

            if k % MAXS == 0 and k != 0:
                X = np.array(X)
                Y = np.array(Y)
                save(dirS + nombreData + '-' + str(int(k/MAXS)) + '-X.mat', X)
                save(dirS + nombreData + '-' + str(int(k/MAXS)) + '-Y.mat', Y)
                X, Y = [], []

            name = img.split('/')[-1]
            img = cv2.imread(img)
            tam = img.shape[0] * img.shape[1]

            try:
                boxs = boxsByName(boundboxs, name)
            except (IndexError, cv2.error):
                continue

            _, R = selective_search(img, colour_space='rgb', scale=SCALA,
                                    sim_threshold=STHSLD)

            propuestas = [procesar(r) for r in R if area(r) < (IGNORAR*tam)]

            for bb in boxs:
                appendValue(X, Y, bb['box'], str(bb['class']), img, modelo, w2vec)

            for bb in propuestas:
                ious = [(i, iou(bb, b['box'])) for i, b in enumerate(boxs)]
                ious.sort(key=lambda x: x[1], reverse=True)

                if ious[0][1] > IOU:
                    cls = str(boxs[ious[0][0]]['class'])
                    appendValue(X, Y, bb, cls, img, modelo, w2vec)

                elif 0 < ious[0][1] < BACKGROUND:
                    appendValue(X, Y, bb, '-1', img, modelo, w2vec)

                elif ious[0][1] == 0 and randint(0, PROB) == 0:
                    appendValue(X, Y, bb, '-1', img, modelo, w2vec)

        if X != []:
            X = np.array(X)
            Y = np.array(Y)
            save(dirS + nombreData + '-' + str(int(math.ceil(k/MAXS))) + '-X.mat', X)
            save(dirS + nombreData + '-' + str(int(math.ceil(k/MAXS))) + '-Y.mat', Y)

if __name__ == "__main__":
    main()
