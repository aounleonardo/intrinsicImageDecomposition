""" This module reimplements Keras's DirectoryIterator. Main changes include:
- Merging the parent class Iterator with this given iterators class.
- Disregarding the class keras.preprocessing.image.ImageDataGenerator and thus the image augmentation.
- Changing the directory crawling logic.
"""

# Python std. libs.
import os
import logging
import threading
import multiprocessing
import time

try:
    import queue
except ImportError:
    import Queue as queue

# Misc. 3rd party libs
import numpy as np

# Project files.
from data_manipulation.image_manipulation import loadImagesTiff
from data_manipulation.dataset_manipulation import loadYfromFiles


class IteratorDirsXY(object):
    """ This class implements the data iterator for the case where we have
    data X (images) and labels Y (mesh vertices) which constitute pairs.
    """
    def __init__(self,
                 dirX, subdirsX,
                 dirY, subdirsY,
                 imgShape=(224, 224), gtShape=(363, ),
                 dimOrdering='tensorflow',
                 batchSize=64, shuffle=False, seed=None,
                 yDataField='y'):
        """ Constructor.

        Parameters
        ----------
        dirX
        dirXtgt
        dirY
        imgShape
        dimOrdering
        batchSize
        shuffle
        seed
        follow_links
        """
        self._dirX = dirX
        self._subdirsX = subdirsX
        self._dirY = dirY
        self._subdirsY = subdirsY
        self._imgShape = tuple(imgShape)
        self._batchSize = batchSize
        self._shuffle = shuffle
        self._seed = seed
        self._yDataField = yDataField
        self._batchIndex = 0
        self._totalBatchesSeen = 0
        self._lock = threading.Lock()
        self._dimOrdering = dimOrdering

        if self._dimOrdering == 'tensorflow':
            self._imgShape = self._imgShape + (3,)
        else:
            self._imgShape = (3,) + self._imgShape
        self._gtShape = gtShape

        whiteListFormatsX = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
        whiteListFormatsY = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

        self._numSamplesX = 0
        self._numSamplesY = 0

        self._filesX = []
        self._filesY = []

        for subdir in self._subdirsX:
            files = os.listdir(os.path.join(self._dirX, subdir))
            for f in files:
                for ext in whiteListFormatsX:
                    if f.lower().endswith('.' + ext):
                        self._numSamplesX += 1
                        self._filesX.append(os.path.join(subdir, f))

        for subdir in self._subdirsY:
            files = os.listdir(os.path.join(self._dirY, subdir))
            for f in files:
                for ext in whiteListFormatsY:
                    if f.lower().endswith('.' + ext):
                        self._numSamplesY += 1
                        self._filesY.append(os.path.join(subdir, f))

        self._filesX = np.array(self._filesX)
        self._filesY = np.array(self._filesY)

        if self._numSamplesX != self._numSamplesY:
            raise Exception('Number of samples in X source and Y source '
                            'datasets does not match, {nx} != {ny}'.format(
                nx=self._numSamplesX, ny=self._numSamplesY))

        logging.info('Found {n} samples in X, Y datasets.'.format(n=self._numSamplesX))

        self.indexGenerator = self._flowIndex(self._numSamplesX, self._batchSize,
                                              self._shuffle, self._seed)

    def reset(self):
        self._batchIndex = 0

    def _flowIndex(self, N, batchSize, shuffle=False, seed=None):
        # Ensure self._batchIndex is 0.
        self.reset()

        while 1:
            if seed is not None:
                np.random.seed(seed + self._totalBatchesSeen)

            # Create batch indices for source and target data.
            if self._batchIndex == 0:
                indexArray = np.arange(N)
                if shuffle:
                    indexArray = np.random.permutation(N)

            currentIndex = (self._batchIndex * batchSize) % N
            if N >= currentIndex + batchSize:
                currentBatchSize = batchSize
                self._batchIndex += 1
            else:
                currentBatchSize = N - currentIndex
                self._batchIndex = 0
            self._totalBatchesSeen += 1

            yield (indexArray[currentIndex: currentIndex + currentBatchSize],
                   currentIndex, currentBatchSize)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        with self._lock:
            indexArray, curIdx, curBS = next(self.indexGenerator)

        # Loading/normalization/tf of images is not under thread lock so it can be done in parallel.
        fPathsX = [os.path.join(self._dirX, f) for f in self._filesX[indexArray]]
        fPathsY = [os.path.join(self._dirY, f) for f in self._filesY[indexArray]]

        batchX = loadImagesTiff(fPathsX, imgDimOrdering=self._dimOrdering, normalize=True)
        batchY = loadImagesTiff(fPathsY, imgDimOrdering=self._dimOrdering, normalize=True)
        # batchY = loadYfromFiles(fPathsY, dataField=self._yDataField)

        return batchX, batchY

    def getNumSamples(self):
        return self._numSamplesX


def generatorQueue(generator, maxQueSize=10, waitTime=0.05, numWorker=1, pickleSafe=False):
    """Builds a queue out of a data generator.
    If pickleSafe, use a multiprocessing approach. Else, use threading.
    """
    generatorThreads = []
    if pickleSafe:
        q = multiprocessing.Queue(maxsize=maxQueSize)
        _stop = multiprocessing.Event()
    else:
        q = queue.Queue()
        _stop = threading.Event()

    try:
        def dataGeneratorTask():
            while not _stop.is_set():
                try:
                    if pickleSafe or q.qsize() < maxQueSize:
                        generatorOutput = next(generator)
                        q.put(generatorOutput)
                    else:
                        time.sleep(waitTime)
                except Exception:
                    _stop.set()
                    raise

        for i in range(numWorker):
            if pickleSafe:
                # Reset random seed else all children processes
                # share the same seed.
                np.random.seed()
                thread = multiprocessing.Process(target=dataGeneratorTask)
            else:
                thread = threading.Thread(target=dataGeneratorTask)
            generatorThreads.append(thread)
            thread.daemon = True
            thread.start()
    except:
        _stop.set()
        if pickleSafe:
            # Terminate all daemon processes.
            for p in generatorThreads:
                if p.is_alive():
                    p.terminate()
            q.close()
        raise

    return q, _stop, generatorThreads
