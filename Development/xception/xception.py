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
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

# Model Parameters
batch_size_pre  = 64
batch_size_fine = 32
epochs_fine = 200

def build_model():
    # Initialize Model
    model = Xception(include_top=False, weights='imagenet')

    # Append Model Output
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(8192, activation='relu')(x)

    # Build Model
    preds = Dense(data.CAT_SIZE, activation='softmax')(x)
    model = Model(inputs=model.inputs, outputs=preds)

    # # first: train only the top layers
    for layer in base_model.layers: layer.trainable = False

    return model

# Initialize Callback Methods
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Initialize Model
model = build_model()

# Compile Model
model.compile(optimizer=Adam(lr=1.0e-4), loss=categorical_crossentropy, metrics=['accuracy'], callbacks=[tensorboard, checkpoint])

# Perform Learning
# TODO: Append System Callback Mechanism to Keep Track of Training
history = model.fit_generator(
    generator = data.data_generator(data.TRAIN_BSON_PATH, batch_size=batch_size_pre),
    steps_per_epoch = data.NUM_TRAIN_PROD / batch_size_fine,
    epochs = epochs_fine,
    verbose = 1
)

# model.save(os.path.join('./', 'xception.h5'))
