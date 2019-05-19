import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt




def main():
    data = keras.datasets.fashion_mnist
    #loading data
    (train_images, train_labels),(test_images,test_labels) = data.load_data()
    #Declaring a list of class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    #shrinking data by dividing the data by 255 making all values less than one
    train_images = train_images/255
    test_images = test_images/255
    print(train_images[2])

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation="relu"),
        keras.layers.Dense(10,activation="softmax")
    ])

    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics = ["accuracy"])

    model.fit(train_images,train_labels,epochs=5)\

    test_loss,test_acc = model.evaluate(test_images,test_labels)

    prediction = model.predict(test_images)
    print("Tested ACC:", test_acc)

    for i in range(5):
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel("Actual : "+class_names[test_labels[i]])
        plt.title("Prediction ->"+class_names[np.argmax(prediction[i])])
        plt.show()

    print(class_names[np.argmax(prediction[0])])


if __name__ == "__main__":
    main()