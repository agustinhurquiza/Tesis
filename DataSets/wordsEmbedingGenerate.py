#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Explanations will be provided later.
# Escrito por Agustin Urquiza
# Contacto: agustin.h.urquiza@gmail.com
# --------------------------------------------------------

import numpy as np
import argparse
import os
import json
import codecs
from gensim.models.keyedvectors import KeyedVectors


def parser():
    """ Funcion encargada de solicitar los argumentos de entrada.

        Returns:
            <class 'argparse.Namespace'>: Argumentos ingresados por el usuario.
    """

    # Argumentos de entrada permitidos.
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True,
                        help="""Modelo pre entrenado.""")

    parser.add_argument('-i', '--palabras', type=str, required=True,
                        help="""Palabras que se quiere obtner el word vector.""")

    parser.add_argument('-s', '--arc_salida', type=str, required=True,
                        help="""Archivo de salida.""")

    args = parser.parse_args()

    return args


def get_word_vector(model, palabra):
    """  Función para obtener la word vector del modelo. Caso de no existir
         un word vector, retorna el vector zero.

        args:
            model: Modelo pre entrenado.
            palabra (String): Palabra que se le quiere obtener el word vector.
        Returns:
            array : word vector de la palabra ingresada.

    """
    try:
        palabra = palabra.lower()
        wv = model[palabra]
        wv.setflags(1)
        return wv
    except KeyError:
        return np.zeros(300)


def main():
    args = parser()

    model_path = args.model
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    f = open(args.palabras, 'r')
    palabras = json.load(f)
    f.close()
    resultado = {}

    for j, (i, palabra) in enumerate(palabras.items()):
        resultado[i] = get_word_vector(model, palabra).tolist()	
        print(str(j+1) + " de: " + str(len(palabras)))

    json.dump(resultado, codecs.open(args.arc_salida, 'w', encoding='utf-8'),
              separators=(',', ':'), indent=2)


if __name__ == "__main__":
    main()
