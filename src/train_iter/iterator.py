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
from scipy import misc

class IteratorDirsXY(object):
    """ This class implements the data iterator for the case where we have
    data I (images), S (shadings) and A (albedos).
    """
    def __init__(self,
                 dirI,dirS,dirA, subdirs,
                 imgShape=(224, 224),
                 dimOrdering='tensorflow',
                 batchSize=64, shuffle=False, seed=None):
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
        self._dirI = dirI
        self._dirS = dirS
        self._dirA = dirA
        self._subdirs = subdirs
        self._imgShape = tuple(imgShape)
        self._batchSize = batchSize
        self._shuffle = shuffle
        self._seed = seed
        self._batchIndex = 0
        self._totalBatchesSeen = 0
        self._lock = threading.Lock()
        self._dimOrdering = dimOrdering

        if self._dimOrdering == 'tensorflow':
            self._imgShape = self._imgShape + (3,)
        else:
            self._imgShape = (3,) + self._imgShape

        whiteListFormats = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

        self._numSamplesI = 0
        self._numSamplesS = 0
        self._numSamplesA = 0

        self._filesI = []
        self._filesS = []
        self._filesA = []

        for subdir in self._subdirs:
            files = os.listdir(os.path.join(self._dirI, subdir))
            for f in files:
                for ext in whiteListFormats:
                    if f.lower().endswith('.' + ext) and not f.lower().startswith('._sh'):
                        self._numSamplesI += 1
                        self._filesI.append(os.path.join(subdir, f))

            files = os.listdir(os.path.join(self._dirS, subdir))
            for f in files:
                for ext in whiteListFormats:
                    if f.lower().endswith('.' + ext) and not f.lower().startswith('._sh'):
                        self._numSamplesS += 1
                        self._filesS.append(os.path.join(subdir, f))

            files = os.listdir(os.path.join(self._dirA, subdir))
            for f in files:
                for ext in whiteListFormats:
                    if f.lower().endswith('.' + ext) and not f.lower().startswith('._sh'):
                        self._numSamplesA += 1
                        self._filesA.append(os.path.join(subdir, f))

        self._filesI = np.array(self._filesI)
        self._filesS = np.array(self._filesS)
        self._filesA = np.array(self._filesA)

        if self._numSamplesI != self._numSamplesS or self._numSamplesI != self._numSamplesA:
            raise Exception('Number of samples in I, S and A sources '
                            'datasets does not match, {ni} != {ns} != {na}'.format(
                ni=self._numSamplesI, ns=self._numSamplesS, na=self._numSamplesA))

        logging.info('Found {n} samples in I, S and A datasets.'.format(n=self._numSamplesI))

        self.indexGenerator = self._flowIndex(self._numSamplesI, self._batchSize,
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
        fPathsI = [os.path.join(self._dirI, f) for f in self._filesI[indexArray]]
        fPathsS = [os.path.join(self._dirS, f) for f in self._filesS[indexArray]]
        fPathsA = [os.path.join(self._dirA, f) for f in self._filesA[indexArray]]

        batchI = loadImagesTiff(fPathsI, normalize=True)
        batchS = loadImagesTiff(fPathsS, normalize=True)
        batchA = loadImagesTiff(fPathsA, normalize=True)

        return batchI, batchS, batchA

    def getNumSamples(self):
        return self._numSamplesI


def loadImagesTiff(paths, normalize=True):
    nb_imgs = len(paths)
    imgs = np.empty([nb_imgs, 224, 224, 3])

    for index in range(nb_imgs):
        imgs[index] = misc.imread(paths[index], mode='RGB') / 255.0

    return imgs

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
