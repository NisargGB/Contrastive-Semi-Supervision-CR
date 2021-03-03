# -*- coding: utf-8 -*-
"""train_anti_celiac.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BpuYFF1SGAfv4XYgDUFAqqj3HEfTGbwN
"""

import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from chaos_data_generator import CHAOS_Loader as DataGenerator
from model import attention_unet_refined, attention_unet_resnet50
from metrics import *
import time
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

## Path
model_path = '/scratch/cse/btech/cs1170354/BTP2/weights/chaos_weights_samesplit.h5' # Saving location
log_path = '/scratch/cse/btech/cs1170354/BTP2/logs/chaos_weights_samesplit'
weights_paths = []
#                  '/content/gdrive/My Drive/Study/BTP/Anti-celiac/weights_train_C-V-E-B-IR_focalloss_0.02_bifmse_1.0.h5',  # Loading location
#                  '/content/gdrive/My Drive/Study/BTP/Anti-celiac/weights.h5']  # Loading location
# encoder_weights = '/content/gdrive/My Drive/Study/BTP/Context_restoration/weights320.h5'    # Weights specifically to initialise the encoder
encoder_weights = None


train_path = "/scratch/cse/btech/cs1170354/BTP2/Data/CHAOS_train/MR"
# train_path = "D:/Master/Study/Semester7/BTP1/Data/Train"

## Parameters
image_size = (256, 256) # Original = (2448, 1920)
batch_size = 16
mode = 'seg'
target_classes = ["Liver", "Left kidney", "Right kidney", "Spleen"]
filter_classes = []    # Train only with items containing them
freeze_encoder = False
freeze_decoder = False
load_last_layer = False
lr = 1e-3
epochs = 500

# att_unet = attention_unet_resnet50(input_shape=(image_size[0], image_size[1], 3)
#                                 , out_channels=len(target_classes)
#                                 , freeze_encoder=freeze_encoder
#                                 , encoder_weights=encoder_weights
#                                 , freeze_decoder=freeze_decoder
#                                 , dropout_rate=0.0)
encoder, att_unet, localizer, anticeliac, mask_inputs = attention_unet_refined(image_size, 
                                                                                3, 
                                                                                len(target_classes), 
                                                                                multiplier=8, 
                                                                                freeze_encoder=freeze_encoder, 
                                                                                freeze_decoder=freeze_decoder, 
                                                                                use_constraints = False, 
                                                                                dropout_rate=0.0)
if mode == 'seg':
    model = att_unet
    metrics = [uniclass_dice_coeff_0, uniclass_dice_coeff_1, uniclass_dice_coeff_2, uniclass_dice_coeff_3, multiclass_dice_coeff]
    losses = multiclass_dice_loss(loss_scales=[1., 1., 1., 1.])
    # losses = focal_tversky_loss

optimizer = Adam(learning_rate=lr)
model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
model.summary()

# Resume from checkpoint
if weights_paths != []:
    for wp in weights_paths:
        if load_last_layer:
            model.load_weights(wp, by_name=True)
        else:
            _, temp_model, _, _, _ = attention_unet_refined(image_size, 
                                                            3, 
                                                            1, 
                                                            multiplier=10, 
                                                            freeze_encoder=freeze_encoder, 
                                                            freeze_decoder=freeze_decoder, 
                                                            use_constraints = False, 
                                                            dropout_rate=0.0)
            temp_model.load_weights(wp, by_name=True)
            for i in range(len(model.layers)):
                if model.layers[i].name != 'conv1x1':
                    model.layers[i].set_weights(temp_model.layers[i].get_weights())


# Data generators
image_ids = []
for patient in os.listdir(train_path):
    dir_pth = os.path.join(train_path, patient, 'T2SPIR', 'DICOM_anon')
    imgs = os.listdir(dir_pth)
    imgs = [os.path.join(dir_pth, img) for img in imgs]
    image_ids.extend(imgs)
random.shuffle(image_ids)
num_imgs = len(image_ids)
train_image_ids = image_ids[:(7 * num_imgs // 10)]
val_image_ids = image_ids[(7 * num_imgs // 10):]
train_generator = DataGenerator(train_image_ids, image_size, batch_size, mode, target_classes, filter_classes=filter_classes, augment=True)
val_generator = DataGenerator(val_image_ids, image_size, batch_size, mode, target_classes, augment=False)


callbacks = [ModelCheckpoint(model_path, save_weights_only=True),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor='loss', patience=25, restore_best_weights=False), 
            TensorBoard(log_dir=log_path, update_freq='epoch', write_graph=False, profile_batch=0),
]

model.fit_generator(generator = train_generator
                , steps_per_epoch = train_generator.__len__()
                , epochs = epochs
                , verbose = 1
                , callbacks = callbacks
                , validation_data = val_generator
                , validation_steps = val_generator.__len__()
                , validation_freq = 2
                , workers = 8
                , max_queue_size = 20
                , initial_epoch = 0)
