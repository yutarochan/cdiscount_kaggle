'''
CDiscount: Xception Model (No Pretrained)
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
import sys
from keras.models import Model, load_model
from keras.applications.xception import Xception

sys.path.append('..')
import util.data as data

# Model Parameters
batch_size_fine = 256
batch_size_pre  = 128
epochs_fine = 100

def build_model():
    # Initialize Model
    model = Xception(include_top=False, weights=None)

    # Append Model Output
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(8192, activation='relu')(x)

    # Build Model
    preds = Dense(data.CAT_SIZE, activation='softmax')(x)
    model = Model(inputs=model.inputs, outputs=preds)

    return model

# Initialize Model
model = build_model()

# Compile Model
model.compile(optimizer=Adam(lr=1.0e-4), loss=categorical_crossentropy, metrics=['accuracy'])

# Perform Learning
# TODO: Append System Callback Mechanism to Keep Track of Training
history = model.fit_generator(
    generator = data.data_generator(data.TRAIN_BSON_PATH, batch_size=batch_size_pre),
    steps_per_epoch = math.ceil(len(data.NUM_TRAIN_PROD) / batch_size_fine),
    epochs = epochs_fine
    verbose = 1
)

# model.save(os.path.join('./', 'xception.h5'))
