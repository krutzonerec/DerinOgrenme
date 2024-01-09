# -*- coding: utf-8 -*-
"""Derin Ogrenme - 65% Train 35% Test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12017WlbxTcyW42NsIvMzO3tEBf4T8rFS
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from google.colab import drive
import shutil
import numpy as np

from google.colab import drive
drive.mount('/gdrive')

base_path = '/gdrive/My Drive/Kidney_Cancer/Kidney Cancer'
tumor_path = os.path.join(base_path, 'Tumor')
normal_path = os.path.join(base_path, 'Normal')

tumor_files = [os.path.join(tumor_path, f) for f in os.listdir(tumor_path) if os.path.isfile(os.path.join(tumor_path, f))]
normal_files = [os.path.join(normal_path, f) for f in os.listdir(normal_path) if os.path.isfile(os.path.join(normal_path, f))]

data = tumor_files + normal_files
labels = ['tumor'] * len(tumor_files) + ['normal'] * len(normal_files)

X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.35, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(10/35), random_state=42)

train_datagen = ImageDataGenerator(rescale=1./255)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#burada modelin hizli egitilmesi icin optimizer belirliyoruz
#bu cok sagliksiz bir sey bence ama yine de vakit az oldugu icin learning rate bu sekilde
optimizer = Adam(learning_rate=0.001)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# burada val loss ve test loss arasinda ucurum olusup da overfitting olmamasi adina early stopping belirliyoruz
#fit ederken
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#modeli fit ediyoruz
history = model.fit(train_datagen.flow_from_directory(base_path, target_size=(32, 32), class_mode='binary', batch_size=64),
                    validation_data=val_test_datagen.flow_from_directory(base_path, target_size=(32, 32), class_mode='binary', batch_size=64),
                    epochs=10,
                    callbacks=[lr_schedule, early_stopping])

#bunlarin neden 1 ciktigini bilmiyorum sanki 1 olmamali gibi bir his var
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

print("Training Accuracy: ", train_acc[-1])
print("Validation Accuracy: ", val_acc[-1])

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# dogruluk oranlarini yazdiriyoruz
for epoch, (train_acc, val_acc) in enumerate(zip(train_accuracy, val_accuracy), 1):
    print(f"Epoch {epoch}: Training Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}")



"""## TRANSFER LEARNING VGG ALGROITMASI"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

#VGG16 algoritmasini model olarak aliyoruz

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
vggmodel = Model(inputs=base_model.input, outputs=predictions)


vggmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#kendi modelimizdeki gibi burada da early stopping beirliyoruz
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

history = model.fit(
    train_datagen.flow_from_directory(base_path, target_size=(32, 32), class_mode='binary', batch_size=64),
    steps_per_epoch=len(X_train) // 64,
    epochs=5,
    validation_data=val_test_datagen.flow_from_directory(base_path, target_size=(32, 32), class_mode='binary', batch_size=6),
    validation_steps=len(X_val) // 64,
    callbacks=[lr_schedule, early_stopping]
)

train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

"""## KENDI NEURAL NETWORK OLSUTURMA


"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, InputLayer, Input
from tensorflow.keras.optimizers import Adam

input_shape = (32, 32, 1)

x_input = Input(shape=input_shape)

x = Dense(128, activation='relu')(x_input)
x = Dense(64, activation='relu')(x)

output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=x_input, outputs=output)
