# Trains the model using both synthetic data with labels and real data with depth maps. Since the loss on depth maps
# do not preserve the topology of the mesh (we don't have correspondences between the predicted mesh vertices and
# their GT position within the depth map), we need to alternate between using the synth. data which preserve the
# topology and real data which make the network generalize on real data as well.

import matplotlib as mpl

mpl.use('Agg')

# Python
import logging.config
import sys
from timeit import default_timer as timer
import time

# Keras
from keras.optimizers import adam
from keras.callbacks import History
from keras.models import Model
from keras.objectives import MSE
import keras.backend as K

# TF
import tensorflow as tf

# Project files.
from data_manipulation.dataset_manipulation import *
from data_manipulation.mesh_manipulation import loadDatasetInfo2
from utils.io_utils import saveModel
import dnn_architectures.keras.resnet as resnet
from train.objectives import meanVertexPairEuclideanDistanceTF, getDepthMSEObjective, depthMSE_TEST
from utils.path_utils import getDatasetGenerationPaths, getLearningDataPaths
from data_manipulation.iterator import IteratorDirsXY, generatorQueue
from train.callbacks import HistorySaver, WeightsSaver

########################################################################################################################
# Input parameters and hyperparameters
########################################################################################################################

# Optimization framework
FRAMEWORK = 'tensorflow'

# Num mesh vertices.
numOutputVertices = 121
# numOutputVertices = 961

# Size of depth map
shapeDepthMap = (480, 480)

# Relative path to dataset we are dealing with.
dsSynthRelPath = 'synth_white_cloth_3'
dsRealRelPath = 'real_white_cloth_3'

# Training, validation, testing data directories.
# We don't use these directories at all:
# 'brbltl-rubber', 'tltrbr-rubber'
dirsSynthTrain = ['b31_tl_tr-cotton', 'brbltl-leather', 'r31_tl_bl-rubber', 'tltrbrbl-cotton',
            #  'b31_tl_tr-leather', 't31_bl_br-cotton', 'tltrbrbl-leather',
            #  'l31_tr_br-cotton', 't31_bl_br-leather', 'tltrbrbl-rubber',
            #  'bltltr-cotton', 'l31_tr_br-leather', 't31_bl_br-rubber', 'trbrbl-cotton',
            #  'l31_tr_br-rubber', 'tltrbr-cotton', 'trbrbl-leather',
            #  'bltltr-rubber', 'r31_tl_bl-cotton', 'tltrbr-leather', 'trbrbl-rubber',
             'brbltl-cotton', 'r31_tl_bl-leather']
dirsSynthValid = ['b31_tl_tr-rubber', 'bltltr-leather']

dirsRealTrain = ['seq_01', 'seq_02', 'seq_04', 'seq_05', 'seq_06', 'seq_08']
dirsRealValid = ['seq_03']

# Relative paths to X, Y data (w.r.t. ds base path).
relPathSynthImages = 'white_01/img_224x224'
relPathSynthLabels = 'white_01/ctrl_121_range_1'
relPathRealImages = '.'
relPathRealDepthMaps = 'depth_480x480_flat_m'

dataInfoFile = 'data_info2.pkl'

# Camera intrinsic matrix related to depth maps (as seen by Kinect's RGB camera)
focalLength = 1.870866e-3 # Kinect RGB cam, [m]
sensorWidth = 1.728e-3 # Kinect RGB cam, [m]
wPX = shapeDepthMap[1]

f = focalLength / (sensorWidth / wPX)

A = np.array([[f, 0.0, 240.0],
              [0.0, f, 240.0],
              [0.0, 0.0, 1.0]])

# Save flags.
flagSaveModel = False
flagSaveWeights = True
flagSaveHistory = True

#################################################################################################
# HELPERS
#################################################################################################

def loadBatch(que, stop):
    while not stop.is_set():
        if not que.empty():
            genOutput = que.get()
            break
        else:
            time.sleep(0.01)

    if not hasattr(genOutput, '__len__'):
        stop.set()
        raise ValueError('Output of generator should be a tuple '
                         '(X, Y). Found: ' + str(genOutput))

    return genOutput

#################################################################################################
# TRAINING PARAMETERS
#################################################################################################

lr = 1e-3
epochs = 256
batchSize = 32

#################################################################################################
# LOAD DATA
#################################################################################################

# Get all paths to datasets and learning data.
dsSynthPaths = getDatasetGenerationPaths(dsSynthRelPath)
dsRealPaths = getDatasetGenerationPaths(dsRealRelPath)
ldPaths = getLearningDataPaths()

# Prepare datasets dirs.
imgsSynthDir = os.path.join(dsSynthPaths['images'], relPathSynthImages)
labelsSynthDir = os.path.join(dsSynthPaths['labels'], relPathSynthLabels)
imgsRealDir = os.path.join(dsRealPaths['images'], relPathRealImages)
depthMapsRealDir = os.path.join(dsRealPaths['depthMaps'], relPathRealDepthMaps)

# Train dataset generator.
dataGenSynthTr = IteratorDirsXY(imgsSynthDir, dirsSynthTrain,
                           labelsSynthDir, dirsSynthTrain,
                           imgShape=(224, 224), gtShape=(363, ),
                           dimOrdering=FRAMEWORK, batchSize=batchSize,
                           shuffle=True)
dataGenSynthVa = IteratorDirsXY(imgsSynthDir, dirsSynthValid,
                           labelsSynthDir, dirsSynthValid,
                           imgShape=(224, 224), gtShape=(363, ),
                           dimOrdering=FRAMEWORK, batchSize=batchSize,
                           shuffle=False)
dataGenRealTr = IteratorDirsXY(imgsRealDir, dirsRealTrain,
                               depthMapsRealDir, dirsRealTrain,
                               imgShape=(224, 224), gtShape=(480 * 480, ),
                               dimOrdering=FRAMEWORK, batchSize=batchSize,
                               shuffle=True, yDataField='depth')
dataGenRealVa = IteratorDirsXY(imgsRealDir, dirsRealValid,
                               depthMapsRealDir, dirsRealValid,
                               imgShape=(224, 224), gtShape=(480 * 480, ),
                               dimOrdering=FRAMEWORK, batchSize=batchSize,
                               shuffle=True, yDataField='depth')

# Get number of samples.
NSynthTr = dataGenSynthTr.getNumSamples()
NSynthVa = dataGenSynthVa.getNumSamples()
NRealTr = dataGenRealTr.getNumSamples()
NRealVa = dataGenRealVa.getNumSamples()

# debug
print('Num synth. training samples = {}'.format(NSynthTr))
print('Num synth. validation samples = {}'.format(NSynthVa))
print('Num real training samples = {}'.format(NRealTr))
print('Num real validation samples = {}'.format(NRealVa))

# Create data generation queues.
queSynthTr, _stopSynthTr, genThreadsSynthTr = generatorQueue(dataGenSynthTr, maxQueSize=20, numWorker=10)
queSynthVa, _stopSynthVa, genThreadsSynthVa = generatorQueue(dataGenSynthVa, maxQueSize=20, numWorker=1)
queRealTr, _stopRealTr, genThreadsRealTr = generatorQueue(dataGenRealTr, maxQueSize=20, numWorker=10)
queRealVa, _stopRealVa, genThreadsRealVa = generatorQueue(dataGenRealVa, maxQueSize=20, numWorker=1)

# Load custom user tf.
C = loadDatasetInfo2(os.path.join(labelsSynthDir, dataInfoFile), encoding='latin1')[3]

########################################################################################################################
# Training functions for each DNN model
########################################################################################################################

def resn(inputShape, outputSize, depthMapShape, type='resnet34'):
    outName = type + '_synth_and_real_white_cloth_y_{nov}_range1'.format(nov=numOutputVertices)

    # Save data parameters
    fileBestModel = outName + '_model.h5'
    fileHistory = outName + '_history.pkl'
    fileWeights = outName + '_weights.h5'

    pathW = os.path.join(ldPaths['weights'], fileWeights)
    pathM = os.path.join(ldPaths['models'], fileBestModel)
    pathH = os.path.join(ldPaths['histories'], fileHistory)

    # Create model.
    logging.info('Creating model {t}.'.format(t=type))
    modelSynth = resnet.createModel(inputShape, outputSize, type=type, name='synth')
    modelReal = Model([modelSynth.input], [modelSynth.output], name='real')

    optimizerSynth = adam(lr=lr)
    optimizerReal = adam(lr=lr)

    modelSynth.compile(loss=meanVertexPairEuclideanDistanceTF,
                       optimizer=optimizerSynth)

    depthMSE = getDepthMSEObjective(depthMapShape, A, C)
    # depthMSE = depthMSE_TEST

    modelReal.compile(loss=depthMSE,
                      optimizer=optimizerReal)

    # Save model.
    # if flagSaveModel:
    #     logging.info('Saving model: {}'.format(pathM))
    #     saveModel(pathM, model)

    # Create history.
    history = History()
    history.history = {'loss_synth': [], 'loss_real': [], 'val_loss_synth': [], 'val_loss_real': []}
    history.model = None
    modelSynth.history = history

    # Callbacks
    hSaver = HistorySaver(fileName=pathH, period=1)
    wSaver = WeightsSaver(fileName=pathW, period=1)
    hSaver.model = modelSynth
    wSaver.model = modelSynth

    logging.info('Training model.')

    # Training loop
    numBatchesRealTr = int(np.ceil(NRealTr / batchSize))
    numBatchesSynthVa = int(np.ceil(NSynthVa / batchSize))
    numBatchesRealVa = int(np.ceil(NRealVa / batchSize))
    for ep in range(epochs):
        print('Epoch {ep}'.format(ep=ep + 1))

        tStart = timer()
        for it in range(numBatchesRealTr):
            print('Batch: {b}/{tb}.'.format(b=it + 1, tb=numBatchesRealTr), end=' ')

            # Get next synth batch.
            genOutputSynth = loadBatch(queSynthTr, _stopSynthTr)
            batchXtrSynth = genOutputSynth[0]
            batchYtrSynth = genOutputSynth[1]

            # Train on 1 synth batch.
            lossTrSynth = float(modelSynth.train_on_batch(batchXtrSynth, batchYtrSynth))

            # Get next real batch.
            genOutputReal = loadBatch(queRealTr, _stopRealTr)
            batchXtrReal = genOutputReal[0]
            batchYtrReal = genOutputReal[1]

            # Train on 1 real batch.
            lossTrReal = float(modelReal.train_on_batch(batchXtrReal, batchYtrReal))

            print('Training loss (synth/real): {s:.4f}/{r:.4f}.'.format(s=lossTrSynth, r=lossTrReal), end=' ')
            print('t: {:0.2f} s.'.format(timer() - tStart),
                  end=('\r', ' ')[it == numBatchesRealTr - 1])

        # Validation for synth data.
        lossVaSynth = 0.0
        for it in range(numBatchesSynthVa):
            # Get next batch.
            genOutput = loadBatch(queSynthVa, _stopSynthVa)
            batchXva = genOutput[0]
            batchYva = genOutput[1]

            # Validate on 1 batch.
            lossVaSynth += batchXva.shape[0] * \
                float(modelSynth.evaluate(batchXva, batchYva, verbose=0))
        lossVaSynth /= NSynthVa

        # Validation for real data.
        lossVaReal = 0.0
        for it in range(numBatchesRealVa):
            # Get next batch.
            genOutput = loadBatch(queRealVa, _stopRealVa)
            batchXva = genOutput[0]
            batchYva = genOutput[1]

            # Validate on 1 batch.
            lossVaReal += batchXva.shape[0] * \
                           float(modelReal.evaluate(batchXva, batchYva, verbose=0))
        lossVaReal /= NRealVa

        print('Validation loss (synth/real): {s:.4f}/{r:.4f}'.format(s=lossVaSynth, r=lossVaReal))

        # Update history.
        history.history['loss_synth'].append(lossTrSynth)
        history.history['loss_real'].append(lossTrReal)
        history.history['val_loss_synth'].append(lossVaSynth)
        history.history['val_loss_real'].append(lossVaReal)

        # Save weights, history.
        if flagSaveHistory:
            hSaver.on_epoch_end(ep)
        if flagSaveWeights:
            wSaver.on_epoch_end(ep)


########################################################################################################################
# Main script
########################################################################################################################

# Enable pickling and saving the History object.
sys.setrecursionlimit(10000)

# Initialize logging.
logging.config.fileConfig('log/logging.conf')

if FRAMEWORK == 'theano':
    inputShape = (3, 224, 224)
elif FRAMEWORK == 'tensorflow':
    inputShape = (224, 224, 3)
else:
    raise Exception('Unknown framework {}.'.format(FRAMEWORK))

# Set tf session (to log device placement).
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.set_session(sess)

outputSize = numOutputVertices * 3
outputShape = (3, 11, 11)

## ResNet34
resn(inputShape, outputSize, shapeDepthMap, type='resnet34')
