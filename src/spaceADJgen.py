import numpy as np
import cv2 as cv
from numpy.lib.twodim_base import triu_indices_from
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm


def generateSF(imgsize):
    idxlist = np.arange(imgsize)
    colidx = np.reshape(np.tile(idxlist, (imgsize, 1)), (imgsize*imgsize, 1))
    rowidx = np.reshape(np.tile(np.reshape(idxlist, (-1, 1)), imgsize), (imgsize * imgsize, 1))
    idxfeature = np.concatenate([rowidx, colidx], axis=-1)
    return idxfeature


def generateSADJ(imgsize, n_neighbors):
    idxfeature = generateSF(imgsize)
    adj = kneighbors_graph(idxfeature, n_neighbors).toarray()
    adjidx = np.where(adj > 0)[1]
    return adjidx.astype(np.int32)


def generateSagADJ(locate, n_neighbors):
    # idxfeature = generateSF(imgsize)
    adj = kneighbors_graph(locate, n_neighbors).toarray()
    adjidx = np.where(adj > 0)[1]
    return adjidx

def main():

    print(generateSADJ(3,2))



if __name__ == "__main__":
    main()

