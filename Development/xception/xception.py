'''
CDiscount: Xception Model (No Pretrained)
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
import sys
import math

import tensorflow as tf

from keras.models import Model
from keras.optimizers import Adam
from keras.applications.xception import Xception
from keras.losses import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint

sys.path.append('..')
import util.data as data

# Application Parameters
base_model_wp = "base_model-{epoch:02d}-{val_acc:.2f}.hdf5"
fine_model_wp = "fine_model-{epoch:02d}-{val_acc:.2f}.hdf5"

# Model Parameters
batch_size_pre  = 64
batch_size_fine = 32

epochs_pre  = 10
epochs_fine = 200

# Initialize Callback Methods
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
chk_pt_1 = ModelCheckpoint(base_model_wp, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
chk_pt_2 = ModelCheckpoint(fine_model_wp, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Initialize Model
base_model = Xception(include_top=False, weights='imagenet')

# Append Model Output
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(8192, activation='relu')(x)

# Build Model
preds = Dense(data.CAT_SIZE, activation='softmax')(x)
model = Model(inputs=base_model.inputs, outputs=preds)

# Phase 1: Base Model Training
print('[Phase 1]: Base Model Training')
for layer in base_model.layers: layer.trainable = False
model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])
model.fit_generator(
    generator = data.data_generator(data.TRAIN_BSON_PATH, batch_size=batch_size_pre),
    steps_per_epoch = math.ceil(data.NUM_TRAIN_PROD / batch_size_pre),
    callbacks=[tensorboard, chk_pt_1],
    epochs = epochs_pre,
    verbose=1,
)

# Phase 2: Train All Layers (with Lower Learning Rate)
for layer in model.layers: layer.trainable = True
model.compile(optimizer=Adam(lr=1.0e-4), loss=categorical_crossentropy, metrics=['accuracy'])
model.fit_generator(
    generator = data.data_generator(data.TRAIN_BSON_PATH, batch_size=batch_size_fine),
    steps_per_epoch = math.ceil(data.NUM_TRAIN_PROD / batch_size_fine),
    callbacks=[tensorboard, chk_pt_2],
    epochs = epochs_fine,
    verbose=1,
)

model.save(os.path.join('./', 'cdiscount_xception.hdf5'))
