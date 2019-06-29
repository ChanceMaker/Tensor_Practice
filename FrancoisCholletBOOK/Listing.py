#Listing 5.1 Instantiating a small convnet

from keras import layers
from keras import models

from keras.datasets import mnist
from keras.utils import to_categorical

import matplotlib.pyplot as plt
# model = models.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(54,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))

# #model.summary()
# #Listing 5.2 Adding a classifier on top of the convnet

# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(10,activation='softmax'))

#model.summary()
#Listing 5.3 Training the convnet on MNIST Images
#importing datasets and utils from keras in importing the mnist dataset and to_categorical transformation function


(train_images,train_labels),(test_images, test_labels) = mnist.load_data()#loading the data
# train_images = train_images.reshape((60000,28,28,1))
# train_images = train_images.astype('float32')/255 #Make all values between 0 and 1

# test_images = test_images.reshape((10000,28,28,1))
# test_images = test_images.astype('float32')/255

# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)

# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics = ['accuracy'])#configures the model for training
# model.fit(train_images,train_labels,epochs=5,batch_size=64)


# model.save("model.h5")
#now for evaluating the model on the test data
#print("Accuracy of the model : {0} \nLoss of the model {1}".format(test_acc,test_loss))

model = models.load_model("model.h5")

#print("Accuracy of model: {0}\nLoss of model: {1}".format(test_acc,test_loss))
model.summary()

#Listing 2.6
digit = train_images[25]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
