# import lib's
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
# try VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# arrange all the images to the same size
IMAGE_SIZE = [224,224] # choosen 224 because resnet50 demands

train_path = 'Datasets/train'
test_path = 'Datasets/test'

# import resnet50 lib
# add preprocessing layer at front of VGG

resnet = ResNet50(input_shape=IMAGE_SIZE+ [3], weights='imagenet',include_top=False)
# include_top=False --> don't include first and last layer

resnet.summary() # first and last layer ve to add

# don't train existing weights i,e dont train imagenet weights
for layer in resnet.layers:
    layer.trainable = False

# get number of output classes
folders = glob(train_path+'/*') # length of this is the no of categories

# flatten the layers
x = Flatten()(resnet.output)

prediction = Dense(len(folders),activation='softmax')(x)

# model object
resnetModel = Model(inputs=resnet.input,outputs=prediction)

# view the structure of the created model
resnetModel.summary()

# inform resnetModel about cost and optimization to use
resnetModel.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

# lets do data augmentation using ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2
                                   ,horizontal_flip=True)

# for test data, don't ve to data augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

traning_set = train_datagen.flow_from_directory(train_path,
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode = 'categorical')

testing_set = train_datagen.flow_from_directory(test_path,
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode = 'categorical')

# fit the model
modelrun = resnetModel.fit_generator(traning_set,
                                     validation_data=testing_set,
                                     epochs=50,
                                     steps_per_epoch=len(traning_set),
                                     validation_steps=len(testing_set))

'''
Epoch 50/50
2/2 [==============================] - 3s 1s/step - loss: 0.2907 - accuracy: 0.8750 - val_loss: 0.7733 - val_accuracy: 0.7069

'''

modelrun.history

# plot the loss
plt.plot(modelrun.history['loss'], label='train loss')
plt.plot(modelrun.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(modelrun.history['accuracy'], label='train acc')
plt.plot(modelrun.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# save the model i,e .h5 file

from tensorflow.keras.models import load_model

resnetModel.save('carBrandClassificationmodel_reset50.h5')

# lets do prediction of test data
y_pred = resnetModel.predict(testing_set)

'''
[4.1419858e-04, 2.5839682e-03, 9.9700183e-01], probability of classes
'''

# from prediction (y_pred) lets pick up the max value
import numpy as np
y_pred = np.argmax(y_pred,axis=1)

# test for new data

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

loadModel = load_model('carBrandClassificationmodel_reset50.h5')

new_image = image.load_img('Datasets/Test/lamborghini/1.jpg',target_size=(224,224))
# new_image = <PIL.Image.Image image mode=RGB size=224x224 at 0x215B2CB67B8>

# convert new_image to array
new_image_inArray = image.img_to_array(new_image)

new_image_inArray.shape

# earlier for test data we divided images by 255 for scalling the values, same now
new_image_inArray = new_image_inArray/255

# expand an expansion for compatible with model
new_image_inArray = np.expand_dims(new_image_inArray,axis=0)
new_image_inArrayExpandDim = preprocess_input(new_image_inArray)
new_image_inArrayExpandDim.shape

# model prediction
loadModel.predict(new_image_inArrayExpandDim)
# array([[0.00131329, 0.01161345, 0.9870733 ]], dtype=float32)

predictionClass = np.argmax(loadModel.predict(new_image_inArrayExpandDim),axis=1)
# array([2], dtype=int64)
# unluckly wrong prediction


