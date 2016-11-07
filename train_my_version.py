import tensorflow as tf
from models.slim.deployment import model_deploy
from models.slim.preprocessing import preprocessing_factory
from models.slim.datasets import dataset_factory
from tensorflow.python.ops import control_flow_ops

import vgg_cifar10

slim = tf.contrib.slim

experiment_name = 'vgg_pretrain'
DATA_DIR = '/home/ziba/DataSets/cifar10'
TRAIN_DIR = '/home/ziba/WVU/tfResults/vgg_cifar10_results/LR/0.0005 + EXPERIMENT_NAME + '/train'
BATCH_SIZE = 256
INIT_LEARNING_RATE = 0.001
LR_DECAY = 0.99
NUM_EPOCHS_PER_DECAY = 2
MAX_STEPS = 14000
NUM_CLONES = 1
EXCLUDED_SCOPES = ('vgg_16/conv2/conv2_2',
                   'vgg_16/conv3/conv3_3'
                   'vgg_16/conv4/conv4_3',
                   'vgg_16/conv5/conv5_3',
                   'vgg_16/fc6',
                   'vgg_16/fc7',
                   'vgg_16/fc8')
CHECKPOINT_PATH = '/home/ziba/WVU/tfmodels/mine/vgg-cifar10/pretrained_weights/vgg_16.ckpt'


