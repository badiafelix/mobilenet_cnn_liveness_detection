import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ACTIVATE GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num Gpus Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# DATA PATH
train_path = 'data/NUAA/train/'
valid_path = 'data/NUAA/valid'
test_path = 'data/NUAA/test'

train_length = 8000
val_length = 2000

batch_size = 16

train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(train_path,
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    target_size=(224, 224))

VALIDATION_DIR = valid_path
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(valid_path,
                                                              batch_size=batch_size,
                                                              class_mode='binary',
                                                              target_size=(224, 224))

TESTING_DIR = test_path
testing_datagen = ImageDataGenerator(rescale=1.0/255.)
testing_generator = testing_datagen.flow_from_directory(test_path,
                                                              batch_size=10,
                                                              class_mode='binary',
                                                              shuffle=False,
                                                              target_size=(224, 224))



tf.keras.backend.clear_session()

model = Sequential()
model.add(tf.keras.applications.mobilenet.MobileNet(include_top = False, weights="imagenet", input_shape=(224, 224, 3)))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.layers[0].trainable = True

#model.summary()

checkpoint_filepath = 'best_model.h5'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, save_weights_only=True, monitor='val_acc', mode='max')


model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
# RUN TRAIN MODEL
history = model.fit(
                    train_generator,
                    steps_per_epoch=train_length // batch_size,
                    epochs=20,
                    validation_data=validation_generator,
                    validation_steps=val_length // batch_size,
                    callbacks=[model_checkpoint])

model.load_weights(checkpoint_filepath)

#model.save('liveness_detection_Mobilenet_NUAA1.model')
#model.save('liveness_detection_Mobilenet_NUAA2.model')
#model.save('liveness_detection_Mobilenet_NUAA3.model')

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()
plt.show()



# PREDICTION
#testing_generator.classes
predictions = model.predict(testing_generator, batch_size=10)
y_pred = np.where(predictions > 0.5, 1, 0)



# CONFUSION MATRIX
cm = confusion_matrix(testing_generator.classes, y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

testing_generator.class_indices
cm_plot_labels = ['fake', 'real']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
plt.show()
#END CONFUSION MATRIX


