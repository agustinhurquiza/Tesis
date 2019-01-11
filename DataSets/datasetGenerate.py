#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Explanations will be provided later.
# Escrito por Agustin Urquiza
# Contacto: agustin.h.urquiza@gmail.com
# --------------------------------------------------------

import os
import numpy as np
import argparse
from keras.datasets import cifar10
from keras.preprocessing import image as img
import cv2
import json


def parser():
    """ Funcion encargada de solicitar los argumentos de entrada. Estos
        argumentos son utilizados para modificar las imagenes resultantes.

        Returns:
            <class 'argparse.Namespace'>: Argumentos ingresados por el usuario.

    """

    # Argumentos de entrada permitidos.
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--tam_img', nargs='+', type=int, required=True,
                        help="""Tamaño de las imagenes de salida.""")

    parser.add_argument('-c', '--clas_aleat', type=int, default=0,
                        help="""1: Clases vistas e invisible se elegien\
                                aleatoriamente. Predeterminado 0: clases\
                                predefinidas.""")

    parser.add_argument('-v', '--clas_vis', type=int, default=7,
                        help="""Cantidad de clases vistas, tiene que ser mayor\
                                a 1 y menor a 9. Predeterminado: 7.""")

    parser.add_argument('-ne', '--num_entr', type=int, default=2000,
                        help="""Cantiad de imagenes de entrenamiento.\
                                Predeterminado: 2000""")

    parser.add_argument('-nt', '--num_test', type=int, default=400,
                        help="""Cantiad de imagenes de test.\
                                Predeterminado: 400""")

    parser.add_argument('-mi,', '--max_img', type=int, default=4,
                        help="""Cantidad maxima de imagenes cifar10 por imagen\
                                resultantes. Predeterminado: 4.""")

    parser.add_argument('-dsi', '--dir_salida_img', type=str,
                        default='./Imagenes/cifar-zsd/',
                        help="""Path donde se guardan las imagenes\
                                resultantes.""")

    parser.add_argument('-dsb', '--dir_salida_bb', type=str,
                        default='./BoundingBox/',
                        help="""Path donde se guardan los Bounding Box.""")

    parser.add_argument('-dsc', '--dir_salida_cn', type=str,
                        default='./Classes/',
                        help="""Path donde se guardan los nombres de\
                                las clases.""")

    parser.add_argument('-te', '--tipo_edicion', type=str, default='hrv',
                        help="""h:rotacion horizontal, v:rotacion vertical,\
                                r:rescalado aleatorio. Ejemplos: hr, v.\
                                Predeterminado: hrv.""")

    parser.add_argument('-z', '--zoom_img_fondo', type=int, default=12,
                        help="""Cantidad de zomm de que se le aplica a\
                                a la imagen para crear el fodo.\
                                Predeterminado: 12""")

    parser.add_argument('-rx', '--rescalado_x', type=float, default=3.0,
                        help="""Tamaño maximo del rescalado de las\
                                imagenes en x. Predeterminado: 3.""")

    parser.add_argument('-ry', '--rescalado_y', type=float, default=3.0,
                        help="""Tamaño maximo del rescalado de las\
                                imagenes en y. Predeterminado: 3.""")

    args = parser.parse_args()

    # Revisa que los argumentos sean correctos:
    if args.clas_vis > 9 or args.clas_vis < 1:
        print('Error en los argumentos [Cantidad de calses vistas].')
        exit()

    if len(args.tam_img) != 2 or args.tam_img[0] < 32 or args.tam_img[1] < 32:
        print('Error en los argumentos [Tamaño de las imagenes de salida].')
        exit()

    if args.clas_aleat != 0 and args.clas_aleat != 1:
        print('Error en los argumentos [Clases aleatorias = 0 o 1].')
        exit()

    if args.clas_aleat == 0 and args.clas_vis != 7:
        print('Error en los argumentos [Clases vistas predefinida = 7].')
        exit()

    if args.num_test < 1 or args.num_entr < 1:
        print('Error en los argumentos [La cantidad de imagenes de salida].')
        exit()

    if args.max_img < 3:
        print('Error en los argumentos [Maximo de imagenes > 2].')
        exit()

    if args.zoom_img_fondo < 1 or args.zoom_img_fondo > 32:
        print('Error en los argumentos [Zoom de imagen de fondo < 32 y > 1].')
        exit()

    if args.rescalado_x <= 1 or args.rescalado_x*32 > args.tam_img[0]:
        print('Error en los argumentos [Rescalado de las imagenes].')
        exit()

    if args.rescalado_y <= 1 or args.rescalado_y*32 > args.tam_img[1]:
        print('Error en los argumentos [Rescalado de las imagenes].')
        exit()

    # Crea los directorios de salida.
    if not os.path.exists(args.dir_salida_img):
        os.makedirs(args.dir_salida_img)
        os.makedirs(args.dir_salida_img+'test/')
        os.makedirs(args.dir_salida_img+'train/')

    if not os.path.exists(args.dir_salida_bb):
        os.makedirs(args.dir_salida_bb)

    if not os.path.exists(args.dir_salida_cn):
        os.makedirs(args.dir_salida_cn)

    return args


def rotacion_aletoria(imagen, tipo_edicion):
    """ Funcion encargada de editar las imagenes cifar10.

        Args:
            imagen (array): Imagen de entrada que se quiere editar.
            tipo_edicion (str): 'h': Rotacion horizontal.
                                'v': Rotacion vertical.
        Returns:
            array: Imagen editadad aleatoriamente.

    """

    resultado = np.zeros([IMG_CIF_FILS, IMG_CIF_COLS, IMG_CANALES])
    horz = 'h' in tipo_edicion
    vert = 'v' in tipo_edicion

    imagen = imagen.reshape(1, IMG_CIF_FILS, IMG_CIF_COLS, IMG_CANALES)

    datagen = img.ImageDataGenerator(width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     horizontal_flip=horz,
                                     vertical_flip=vert)

    for X in datagen.flow(imagen, batch_size=1):
        resultado = X.reshape(IMG_CIF_FILS, IMG_CIF_COLS, IMG_CANALES)
        break

    return resultado


def generar_data_set(x_set, y_set, N, tipo_edicion, prefijo):
    """ Funcion encargada de generar las nuevas imagenes y sus respectivos
        BoundingBox.

        Args:
            x_set (array): Imagenes cifar10 que se quiere incluir.
            y_set (array): Clase de verdad de las imagenes.
            N (int): Cantidad de imagenes que se desea generar.
            tipo_edicion (str): 'h': Rotacion aleatorio horizontal.
                                'v': Rotacion aleatorio vertical.
                                'r': Rescalado aleatorio de las imagenes.
            prefijo (str): prefijo de las imagenes que se quiere generar.
        Returns:
            X (array): Imagen ya generadas.
            Y (array): BoundingBox de las imagnes generadas.

    """

    X = np.zeros([N, IMG_FILS, IMG_COLS, IMG_CANALES], dtype=np.uint8)
    Y = []

    for i in range(N):
        cant = np.random.randint(MIN_IMG, MAX_IMG)
        boundboxs = {}
        boundboxs['img_name'] = prefijo + str(i) + '.jpg'
        boundboxs['boxs'] = []

        # Asigna un fondo aleatorio a la imagen.
        j = np.random.randint(0, x_set.shape[0])
        X[i] = cv2.resize(x_set[j, :ZOOM, :ZOOM], dsize=(IMG_COLS, IMG_FILS))

        for j in range(cant):
            boundbox = {}
            anch = IMG_CIF_COLS
            alt = IMG_CIF_FILS

            if 'r' in tipo_edicion:
                anch = np.random.randint(IMG_CIF_COLS, RES_X*IMG_CIF_COLS)
                alt = np.random.randint(IMG_CIF_FILS, RES_Y*IMG_CIF_FILS)

            # Elige una imagen aleatoriamente.
            k = np.random.randint(0, x_set.shape[0])
            # Elige una ubicacion en la imagen aleatoriamente
            x = np.random.randint(0, IMG_COLS-anch)
            y = np.random.randint(0, IMG_FILS-alt)

            s_img = rotacion_aletoria(x_set[k], tipo_edicion)
            s_img = cv2.resize(s_img, dsize=(anch, alt))
            X[i, y:y+alt, x:x+anch] = s_img.astype(np.uint8)

            boundbox['box'] = [x, y, x+anch, y+alt]
            boundbox['class'] = int((y_set[k].sum()))

            boundboxs['boxs'].append(boundbox)

        Y.append(boundboxs)

    return X, Y


def main():
    args = parser()
    # Constantes.
    global IMG_CIF_COLS, IMG_CIF_FILS, IMG_CANALES, NUM_CLASES,\
        NUM_CLASES_VISTA, NUM_CLASES_INVIS, IMG_COLS, IMG_FILS, NUM_ENTR,\
        NUM_TEST, MIN_IMG, MAX_IMG, ZOOM, RES_X, RES_Y

    IMG_CIF_COLS = 32
    IMG_CIF_FILS = 32
    IMG_CANALES = 3
    NUM_CLASES = 10
    NUM_CLASES_VISTA = args.clas_vis
    NUM_CLASES_INVIS = NUM_CLASES - NUM_CLASES_VISTA
    IMG_COLS = args.tam_img[0]
    IMG_FILS = args.tam_img[1]
    NUM_ENTR = args.num_entr
    NUM_TEST = args.num_test
    MIN_IMG = 2
    MAX_IMG = args.max_img
    PATH_I = args.dir_salida_img
    PATH_B = args.dir_salida_bb
    PATH_C = args.dir_salida_cn
    ZOOM = args.zoom_img_fondo
    RES_X = args.rescalado_x
    RES_Y = args.rescalado_y

    LABELS = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    if args.clas_aleat == 1:
        CLASES_VISTA = np.random.choice([i for i in range(NUM_CLASES)],
                                        NUM_CLASES_VISTA, False)
        CLASES_INVIS = np.setdiff1d(np.array([i for i in range(NUM_CLASES)]),
                                    CLASES_VISTA)
    else:
        CLASES_VISTA = np.array([0, 1, 2, 3, 4, 6, 8])
        CLASES_INVIS = np.array([5, 9, 7])

    # Carga el data set cifar10
    (x_entr_cif, y_entr_cif), (x_test_cif, y_test_cif) = cifar10.load_data()

    # Elimina clases invisible de los datos de entrenamiento.
    lista_img = np.array([i[0] for i in np.isin(y_entr_cif, CLASES_VISTA)])
    entr_idxs = np.arange(y_entr_cif.shape[0])[lista_img]
    # Deja solo clases invisble en los datos de prueba.
    lista_img = np.array([i[0] for i in np.isin(y_test_cif, CLASES_INVIS)])
    test_idxs = np.arange(y_test_cif.shape[0])[lista_img]

    x_entr_cif = x_entr_cif[entr_idxs, :, :, :]
    y_entr_cif = y_entr_cif[entr_idxs]
    x_test_cif = x_test_cif[test_idxs, :, :, :]
    y_test_cif = y_test_cif[test_idxs]

    # Crea las imagenes sinteticas.
    X_entr, Y_entr = generar_data_set(x_entr_cif, y_entr_cif, NUM_ENTR,
                                      args.tipo_edicion, 'cifar-zsd-train')
    X_test, Y_test = generar_data_set(x_test_cif, y_test_cif, NUM_TEST,
                                      args.tipo_edicion, 'cifar-zsd-test')

    # Guarda todos los datos en archivos.
    for i in range(NUM_ENTR):
        cv2.imwrite(PATH_I + 'train/cifar-zsd-train'+str(i)+'.jpg', X_entr[i])
    for i in range(NUM_TEST):
        cv2.imwrite(PATH_I + 'test/cifar-zsd-test'+str(i)+'.jpg', X_test[i])

    file = open(PATH_C + 'seen-cifar-zsd.json', 'w')
    file.write(json.dumps({str(key): LABELS.pop(key) for key in CLASES_VISTA}))
    file.close()

    file = open(PATH_C + 'unseen-cifar-zsd.json', 'w')
    file.write(json.dumps(LABELS))
    file.close()

    file = open(PATH_B + 'boundbox-cifar-train-zsd.json', 'w')
    file.write(json.dumps(Y_entr))
    file.close()

    file = open(PATH_B + 'boundbox-cifar-test-zsd.json', 'w')
    file.write(json.dumps(Y_test))
    file.close()


if __name__ == "__main__":
    main()
