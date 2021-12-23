import os
import pickle
import keras
import functools
import numpy as np
import tensorflow as tf

from keras import applications
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam

# Structure of data directories expected
# base_dir/
# ├── train/
# |   ├── class1
# |   |   ├── class1_train_img1.jpg
# |   |   ├── class1_train_img2.jpg
# |   |   └── ...
# |   ├── class2
# |   |   ├── class2_train_img1.jpg
# |   |   ├── class2_train_img2.jpg
# |   |   └── ...
# └── validation/
#     ├── class1
#     |   ├── class1_val_img1.jpg
#     |   ├── class1_val_img2.jpg
#     |   └── ...
#     └── ...
#
# Outputs the following files (with user-specified analysis ID prefixes):
#   final_model_save/   Directory containing the best-performing model in the TensorFlow SavedModel format
#   history.pkl         pickle file containing history of training/validation accuracy and loss rates throughout the run
#   predictions.pkl     pickle file containing predictions by model for all validation images
#   confusion.pkl       pickle file containing raw data to generate confusion matrix
#   labels.pkl          pickle file containing all class labels
#
# Note: Weights files can be downloaded at: https://github.com/fchollet/deep-learning-models/releases/


# Get information on available GPUs in system
tf.config.list_physical_devices('GPU')

# Set up top-3 accuracy metric
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'


################################################ USER INPUT BEGINS ###############################################

# Path to base directory containing training/validation data, etc.
base_dir = ''

# Path to directory to which output will be saved (will be created it it doesn't exist)
output_dir = 'output'

# Analysis number (or any other run identifier)
analysis_id = ''

################################################# RUN PARAMETERS #################################################
#   augment                 bool specifying whether data augmentation should be used                             #
#   reg                     bool specifying whether L1/L2 regularization should be used                          #
#   img_width, img_height   width and height in pixels that input images will be resized to                      #
#   batch_size              number of images in each feedforward batch (limited by memory availability)          #
#   epochs                  number of epochs to run training                                                     #
#   lrate                   learning rate                                                                        #
#   adjust_lrate            bool that specifies whether learning rate should be automatically adjusted           #
#   drop                    dropout parameter (= proportion of features to drop)                                 #
#   lmbda                   lambda parameter of L1/L2 regularization                                             #
#   num_feat                number of augmentation 'treatments' to use (options: 2 or 5)                         #
#   num_classes             total number of classes                                                              #
##################################################################################################################
augment = True
reg = False
img_width, img_height = 160,160
batch_size = 32
epochs = 50
lrate = 0.0001
adjust_lrate = True
drop = 0.5
lmbda = 0.01
num_feat = 2
num_classes = 36
################################################ USER INPUT ENDS #################################################


# Set paths for training and validation data
train_data_dir = os.path.join(base_dir,'train')
validation_data_dir = os.path.join(base_dir,'validation')

# Set weights and initialize models depending on chosen CNN
# include_top is False because we want to add change the size of the final fully-connected
# layer to match the number of classes in our specific problem
model = VGG16(include_top=False, input_shape = (img_width, img_height, 3))
layer_freeze = 7

# Now add additional layers for fine-tuning, regularization, dropout, Softmax, etc.
# Freeze early layers (up to layer specified in layer_freeze) while allowing deeper layers
# to remain trainable for fine-tuning
for layer in model.layers[:layer_freeze]:
    layer.trainable = False

x = model.output
x = Flatten()(x)

# L1/L2 regularization
if reg:
    x = Dense(1024,
              activation="relu",
              kernel_regularizer=regularizers.l2(lmbda),
              activity_regularizer=regularizers.l1(lmbda))(x)

x = Dense(1024, activation="relu")(x)

# Dropout
if drop:
    x = Dropout(drop)(x)
    x = Dense(1024, activation="relu")(x)

# Fully-connected layer for classification
predictions = Dense(num_classes, activation="softmax")(x)

# Finally, we connect the input model layers with the output fully-connected layer and compile
model_final = Model(inputs = model.input, outputs = predictions)
model_final.summary()
model_final.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate=lrate), metrics=['accuracy',top3_acc])


# Data augmentation (if set) and data generators to read and process training/validation images
if augment:
    if num_feat == 5:
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                            rotation_range = 20,
                                            width_shift_range = 0.2,
                                            height_shift_range = 0.2,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip = True,
                                            fill_mode = 'nearest')
    else:
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                            rotation_range = 20,
                                            zoom_range = 0.2,
                                            fill_mode = 'nearest')
else:
    train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_height, img_width),
        class_mode = "categorical")


# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Set up checkpointing (saves best-performing model weights after each epoch) https://github.com/tensorflow/tensorflow/issues/33163
checkpoint = ModelCheckpoint(os.path.join(output_dir,'analysis_{:s}_checkpoint.h5'.format(analysis_id)),
                             monitor='val_accuracy',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             save_freq='epoch')


# Set up early stopping monitor (will stop run if validation accuracy doesn't improve for 10 epochs)
early = EarlyStopping(monitor='val_accuracy',
                      min_delta=0,
                      patience=10,
                      verbose=1,
                      mode='auto')


# Set up automatic learning rate adjustment if requested
if adjust_lrate:
    reduceLR = ReduceLROnPlateau(monitor = 'val_accuracy',factor=0.5,
                                   patience = 3, verbose = 1, mode = 'auto',
                                   min_delta = 0.005, min_lr = 0.00001)
    callbacks = [checkpoint, early, reduceLR]
else:
    callbacks = [checkpoint, early]


# Run model training using generator
history = model_final.fit(train_generator, epochs=epochs, validation_data = validation_generator, callbacks=callbacks) 

# Save best-performing model, in the newer "SavedModel" format. More info: https://www.tensorflow.org/guide/keras/save_and_serialize#savedmodel_format 
model_final.save('final_model_save')

# Save histories
with open(os.path.join(output_dir,'analysis_{:s}_history.pkl'.format(analysis_id)), 'wb') as f:
    pickle.dump(history.history,f)


# Save confusion matrix, classification report, and label map
Y_pred = model_final.predict_generator(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
confusion = confusion_matrix(validation_generator.classes, y_pred)
label_map = (validation_generator.class_indices)
labels = sorted(label_map.keys())
report = classification_report(validation_generator.classes, y_pred, target_names=labels)


with open(os.path.join(output_dir,'analysis_{:s}_predictions.pkl'.format(analysis_id)),'wb') as handle:
    pickle.dump(report,handle)
with open(os.path.join(output_dir,'analysis_{:s}_confusion.pkl'.format(analysis_id)),'wb') as handle:
    pickle.dump(confusion,handle)
with open(os.path.join(output_dir,'analysis_{:s}_labels.pkl'.format(analysis_id)),'wb') as handle:
    pickle.dump(label_map,handle)
