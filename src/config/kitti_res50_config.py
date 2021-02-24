# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""
# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Base Model configurations"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def base_model_config(dataset='PASCAL_VOC'):
  assert dataset.upper()=='PASCAL_VOC' or dataset.upper()=='KITTI', \
      'Currently only support PASCAL_VOC or KITTI dataset'

  cfg = edict()

  # Dataset used to train/val/test model. Now support PASCAL_VOC or KITTI
  cfg.DATASET = dataset.upper()

  if cfg.DATASET == 'PASCAL_VOC':
    # object categories to classify
    cfg.CLASS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                       'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                       'sofa', 'train', 'tvmonitor')
  elif cfg.DATASET == 'KITTI':
    cfg.CLASS_NAMES = ('car', 'pedestrian', 'cyclist')

  # number of categories to classify
  cfg.CLASSES = len(cfg.CLASS_NAMES)    

  # ROI pooling output width
  cfg.GRID_POOL_WIDTH = 7

  # ROI pooling output height
  cfg.GRID_POOL_HEIGHT = 7

  # parameter used in leaky ReLU
  cfg.LEAKY_COEF = 0.1

  # Probability to keep a node in dropout
  cfg.KEEP_PROB = 0.5

  # image width
  cfg.IMAGE_WIDTH = 224

  # image height
  cfg.IMAGE_HEIGHT = 224

  # anchor box, array of [cx, cy, w, h]. To be defined later
  cfg.ANCHOR_BOX = []

  # number of anchor boxes
  cfg.ANCHORS = len(cfg.ANCHOR_BOX)

  # number of anchor boxes per grid
  cfg.ANCHOR_PER_GRID = -1

  # batch size
  cfg.BATCH_SIZE = 20

  # Only keep boxes with probability higher than this threshold
  cfg.PROB_THRESH = 0.005

  # Only plot boxes with probability higher than this threshold
  cfg.PLOT_PROB_THRESH = 0.5

  # Bounding boxes with IOU larger than this are going to be removed
  cfg.NMS_THRESH = 0.2

  # Pixel mean values (BGR order) as a (1, 1, 3) array. Below is the BGR mean
  # of VGG16
  cfg.BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])

  # loss coefficient for confidence regression
  cfg.LOSS_COEF_CONF = 1.0

  # loss coefficient for classification regression
  cfg.LOSS_COEF_CLASS = 1.0

  # loss coefficient for bounding box regression
  cfg.LOSS_COEF_BBOX = 10.0
                           
  # reduce step size after this many steps
  cfg.DECAY_STEPS = 10000

  # multiply the learning rate by this factor
  cfg.LR_DECAY_FACTOR = 0.1

  # learning rate
  cfg.LEARNING_RATE = 0.005

  # momentum
  cfg.MOMENTUM = 0.9

  # weight decay
  cfg.WEIGHT_DECAY = 0.0005

  # wether to load pre-trained model
  cfg.LOAD_PRETRAINED_MODEL = True

  # path to load the pre-trained model
  cfg.PRETRAINED_MODEL_PATH = ''

  # print log to console in debug mode
  cfg.DEBUG_MODE = False

  # a small value used to prevent numerical instability
  cfg.EPSILON = 1e-16

  # threshold for safe exponential operation
  cfg.EXP_THRESH=1.0

  # gradients with norm larger than this is going to be clipped.
  cfg.MAX_GRAD_NORM = 10.0

  # Whether to do data augmentation
  cfg.DATA_AUGMENTATION = False

  # The range to randomly shift the image widht
  cfg.DRIFT_X = 0

  # The range to randomly shift the image height
  cfg.DRIFT_Y = 0

  # Whether to exclude images harder than hard-category. Only useful for KITTI
  # dataset.
  cfg.EXCLUDE_HARD_EXAMPLES = True

  # small value used in batch normalization to prevent dividing by 0. The
  # default value here is the same with caffe's default value.
  cfg.BATCH_NORM_EPSILON = 1e-5

  # number of threads to fetch data
  cfg.NUM_THREAD = 4

  # capacity for FIFOQueue
  cfg.QUEUE_CAPACITY = 100

  # indicate if the model is in training mode
  cfg.IS_TRAINING = False

  return cfg


def kitti_res50_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('KITTI')

  mc.IMAGE_WIDTH           = 1242
  mc.IMAGE_HEIGHT          = 375
  mc.BATCH_SIZE            = 20

  mc.WEIGHT_DECAY          = 0.0001
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 10000
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9
  mc.LR_DECAY_FACTOR       = 0.5

  mc.LOSS_COEF_BBOX        = 5.0
  mc.LOSS_COEF_CONF_POS    = 75.0
  mc.LOSS_COEF_CONF_NEG    = 100.0
  mc.LOSS_COEF_CLASS       = 1.0

  mc.PLOT_PROB_THRESH      = 0.4
  mc.NMS_THRESH            = 0.4
  mc.PROB_THRESH           = 0.005
  mc.TOP_N_DETECTION       = 64

  mc.DATA_AUGMENTATION     = True
  mc.DRIFT_X               = 150
  mc.DRIFT_Y               = 100
  mc.EXCLUDE_HARD_EXAMPLES = False

  mc.ANCHOR_BOX            = set_anchors(mc)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 9

  return mc

def set_anchors(mc):
  H, W, B = 24, 78, 9
  anchor_shapes = np.reshape(
      [np.array(
          [[  94.,  49.], [ 225., 161.], [ 170.,  91.],
           [ 390., 181.], [  41.,  32.], [ 128.,  64.],
           [ 298., 164.], [ 232.,  99.], [  65.,  42.]])] * H * W,
      (H, W, B, 2)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors
