import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
#%matplotlib inline 

import tensorflow as tf
import random
from cv2 import resize
from glob import glob
#from keras.utils.vis_utils import plot_model

import warnings
warnings.filterwarnings("ignore")
def process():
    img_height = 244
    img_width = 244
    train_ds = tf.keras.utils.image_dataset_from_directory('./Dataset/Faulty_solar_panel',validation_split=0.2,subset='training',image_size=(img_height, img_width),batch_size=32,seed=42,shuffle=True)
    val_ds = tf.keras.utils.image_dataset_from_directory('./Dataset/Faulty_solar_panel',validation_split=0.2,subset='validation',image_size=(img_height, img_width),batch_size=32,seed=42,shuffle=True)
    class_names = train_ds.class_names
    print(class_names)
    train_ds
    plt.figure(figsize=(15, 15))
    for images, labels in train_ds.take(1):
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    base_model = tf.keras.applications.VGG16(include_top=False,weights='imagenet',input_shape=(img_height, img_width, 3))
    base_model.trainable = False 
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(90)(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    #plot_model(model, to_file='cnn_plot.png', show_shapes=True, show_layer_names=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    epoch = 15
    model.fit(train_ds, validation_data=val_ds, epochs=epoch, callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=1e-2,patience=3,verbose=1,restore_best_weights=True)])
    base_model.trainable = True
    for layer in base_model.layers[:14]:
        layer.trainable = False
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    epoch = 15
    history = model.fit(train_ds, validation_data=val_ds, epochs=epoch, callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=1e-2,patience=3,verbose=1,)])
    model.save("solar.h5")
    get_ac = history.history['accuracy']
    get_los = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    epochs = range(len(get_ac))
    plt.plot(epochs, get_ac, 'g', label='Accuracy of Training data')
    plt.plot(epochs, get_los, 'r', label='Loss of Training data')
    plt.title('Training data accuracy and loss')
    plt.legend(loc=0)
    plt.figure()
    plt.plot(epochs, get_ac, 'g', label='Accuracy of Training Data')
    plt.plot(epochs, val_acc, 'r', label='Accuracy of Validation Data')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.plot(epochs, get_los, 'g', label='Loss of Training Data')
    plt.plot(epochs, val_loss, 'r', label='Loss of Validation Data')
    plt.title('Training and Validation Loss')
    plt.legend(loc=0)
    plt.figure()
    plt.show()
    loss, accuracy = model.evaluate(val_ds)
    plt.figure(figsize=(20, 20))
    for images, labels in val_ds.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predictions = model.predict(tf.expand_dims(images[i], 0))
            score = tf.nn.softmax(predictions[0])
            if(class_names[labels[i]]==class_names[np.argmax(score)]):
                plt.title("Actual: "+class_names[labels[i]])
                plt.ylabel("Predicted: "+class_names[np.argmax(score)],fontdict={'color':'green'})
            else:
                plt.title("Actual: "+class_names[labels[i]])
                plt.ylabel("Predicted: "+class_names[np.argmax(score)],fontdict={'color':'red'})
                plt.gca().axes.yaxis.set_ticklabels([])        
                plt.gca().axes.xaxis.set_ticklabels([])
    plt.show()
#process()
