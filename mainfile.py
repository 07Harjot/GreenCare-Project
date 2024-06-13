import os
import glob
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D,Activation,AveragePooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


#dataset
train_dir="C://Users//acer//Desktop//GreenCare majorproject//train_set"
test_dir="C://Users//acer//Desktop//GreenCare majorproject//test_set"
def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count=0
    for current_path,dirs,files in os.walk(directory):
        for dr in dirs:
            count+=len(glob.glob(os.path.join(current_path,dr+"/*")))
            return count
    
train_samples=get_files(train_dir)
num_classes=len(glob.glob(train_dir+"/*"))
test_samples=get_files(train_dir)
print(num_classes,"Classes")
print(train_samples,"Train images")
print(test_samples,"Test images")

#preprocessing data
train_datagen=ImageDataGenerator(rescale=1./255,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     validation_split=0.2,
                                     horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)


#set height and width and color of input image
img_width,img_height=256,256
input_shape=(img_width,img_height,3)
batch_size=32
train_generator=train_datagen.flow_from_directory(train_dir,
                                                  target_size=(img_width,img_height),
                                                  batch_size=batch_size)
test_generator=test_datagen.flow_from_directory(test_dir,shuffle=True,
                                                target_size=(img_width,img_height),
                                                batch_size=batch_size)
train_generator.class_indices
#CNN building.
model=Sequential()
model.add(Conv2D(32,(5,5),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))
model.summary()

# model_layers=[layer.name for layer in model.layers]
# print('layer name :',model_layers)

#take one image to visualize it's changes after every layer
# from keras.preprocessing import image
# import keras.utils as image
# import numpy as np
# img1=image.load_img('')
# plt.imshow(img1)
# img1 = image.load_img('',target_size=(256,256))
# img= image.img_to_array(img1)
# img= img/255
# img= np.expand_dims(img, axis=0)
# Visualizing output after every layer. from keras.models import Model
# from keras.models import Model
# conv2d_1_output = Model (inputs=model. input, outputs=model.get_layer('conv2d_3').output)

# max_pooling2d_1_output = Model (inputs=model. input, outputs=model.get_layer ('max_pooling2d_3').output)

# conv2d_2_output = Model (inputs=model.input, outputs=model.get_layer('conv2d_3').output)

# max_pooling2d_2_output = Model (inputs=model. input, outputs=model.get_layer('max_pooling2d_3').output)

# flatten_1_output= Model(inputs=model. input, outputs=model.get_layer('flatten_1').output)

# conv2d_1_features= conv2d_1_output.predict(img)

# max_pooling2d_1_features = max_pooling2d_1_output.predict(img)

# conv2d_2_features = conv2d_2_output.predict(img)

# max_pooling2d_2_features =max_pooling2d_2_output.predict(img)

# flatten_1_features = flatten_1_output.predict(img)




# import matplotlib.image as mping
# fig=plt.figure(figsize=(14,7))
# columns = 8
# rows = 4
# for i in range(columns*rows):
#     #img = mpimg.imread()
#      fig.add_subplot(rows, columns, i+1)
# plt.axis('off')
# plt.title('filter'+str(i))
# plt.imshow(conv2d_1_features [0, :, :, i], cmap='viridis') # Visualizing in color mode.

# plt.show()


#validation data
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size)

#model building to get trained with parameter
from tensorflow import keras
opt=keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
train=model.fit_generator(train_generator,
                          epochs=10,
                          steps_per_epoch=train_generator.samples // batch_size,
                          validation_data=validation_generator,
                          validation_steps=validation_generator.samples// batch_size,verbose=1)



acc = train.history['accuracy']
val_acc = train.history['val_accuracy']
loss = train.history['loss']
val_loss = train.history['val_loss']
epochs = range (1, len (acc) + 1) 
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy') 
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

# save entire model with optimizer, architecture, weights and training configuration
from keras.models import load_model
model.save('greencare.h5')

#save model weights
from keras.models import load_model
model.save_weights('crop_weigths.h5')

#get classes of model trained on
classes= train_generator.class_indices
classes
