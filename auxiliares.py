import numpy as np
import scipy.io as sio
import cv2
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from nms.nms import boxes

def save(filename, mat):
    """ saves data in binary format
    Args:
       filename (str): file name
       mat (ndarray): numpy array
    """
    if not isinstance(mat, np.ndarray):
        raise ValueError('for now, we can only save numpy arrays')
    return sio.savemat(filename, {'data': mat}, appendmat=False)


def load(filename):
    """ load data from file
    Args:
       filename (str): file name

    Returns:
      loaded array
    """
    return sio.loadmat(filename, appendmat=False, squeeze_me=True)['data']


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


def procesar(r):
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

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def predictBox(img, R, unseen, model, resNet, NCOLS=299, NFILS=299, NMSIGNORE=0.2):
    """ Esta funcion obtiene predice las clases al que pertence una lista de
        bounding, ademas se queda con las mejores propuestas. Es responsabiliad
        del usario definir  Y NMSIGNORE.

        Args:
            img (np.array): Imagen a la cual se le quiere obtener las propuesta.
            R (np.array): Todas las propuestas de la imagen.
            unseen (List): Lista de las clases no vistas.
            model (Keras.model): Modelo pre entrenado.
            resNet (Keras.model): Modelo para extraer features de una imagen.
            NCOLS (int): Tamaño de la imagen.
            NFILS (int): Tamaño de la imagen.
            NMSIGNORE (float): Paramatro utilizado para elimianr propuestas similares.

        Returns:
            box_p (List): Una lista con las mejores propuestas y a la clase que
                          pertence cada boundingbox.
    """

    grupos_cls = [[] for _ in range(len(unseen))]
    for bb in R:
        # Pipeline para cada boundingbox propuesta.
        x1, x2, y1, y2 = bb[0], bb[2], bb[1], bb[3]
        x = cv2.resize(img[y1:y2, x1:x2], (NCOLS, NFILS)).reshape(1, NCOLS, NFILS, 3)
        x = resNet.predict(x).reshape(1, -1)
        x = normalize(x, axis=1)
        x = model.predict(x)
        # Descrimina al grupo que pertence la bb segun la similutd coseno.
        x = np.array([2 - cosine(x, cls[1]) for cls in unseen])
        grupos_cls[np.argmax(x)].append((np.max(x), bb))

    box_p = []
    for k, grupo in enumerate(grupos_cls):
        # Non-maximal suppression. Solo quedan las mejores propuestas.
        nms = boxes([i[1] for i in grupo], [i[0] for i in grupo])
        # Elimina las bb similares a la mejor.
        while grupo != [] and nms != []:
            idx = nms[0]
            elm = grupo[idx][1]
            idxs = [k for k, val in enumerate(grupo) if iou(val[1], elm) > NMSIGNORE]
            gruop = [item for i, item in enumerate(grupo) if not i in idxs]
            nms = [item for item in nms if not item in idxs]
            box_p.append((elm, k))

    return box_p
