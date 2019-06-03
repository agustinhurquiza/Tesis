import numpy as np
import scipy.io as sio

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
