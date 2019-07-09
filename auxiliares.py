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


def predictBox(img, R, unseen, model, resNet, NCOLS=299, NFILS=299):
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
        if grupo == []:
            continue
        # Non maximal supression.
        idxs = boxes([i[1] for i in grupo], [i[0] for i in grupo])
        grupo = np.array([[i[1], i[0]]for i in grupo])
        box_p += [(elem[0], k, elem[1]) for elem in grupo[idxs]]

    return box_p


def drawRectangle(img, boxs_t, boxs_p, unseenName):
    """ Esta funicion dibuja los rectangulos en la imagen

        Args:
            img (np.array): Imagen original.
            boxs_t (List): Bounding box verdaderos.
            boxs_p (List): Bounding box propuestos.
            unseen (List): Clases no vistas.
        Returns:
            img_t, img_p: Imagenes con los rectangulos ya dibujados.
    """
    img_p = img.copy()
    img_t = img.copy()
    for b in boxs_t:
        clas = unseenName[str(b[1])]
        x1, x2, y1, y2 = b[0][0], b[0][2], b[0][1], b[0][3]
        cv2.rectangle(img_t, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_t, clas, (x1+10, y1+10), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                    (0, 255, 0), 1)

    for b in boxs_p:
        clas = unseenName[str(b[1])]
        x1, x2, y1, y2 = b[0][0], b[0][2], b[0][1], b[0][3]
        cv2.rectangle(img_p, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_p, clas, (x1+10, y1+10), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                    (255, 0, 0), 1)

    return img_t, img_p
