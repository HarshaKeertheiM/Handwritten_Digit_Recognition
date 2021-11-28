from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2


def main(fname):
    img = prepareImg(cv2.imread(fname), 50)
    img = cv2.bitwise_not(img)
    img = resize(img, (28, 28,1))
    img=np.asarray(img)
    img=img.astype('float32')
    res=model(img)
    return res
def prepareImg(img, height):
	#convert given image to grayscale image (if needed) and resize to desired height
	assert img.ndim in (2, 3)
	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	h = img.shape[0]
	factor = height / h
	return cv2.resize(img, dsize=None, fx=factor, fy=factor)

def model(img):    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    image_index = 7777 # You may select anything up to 60,000
    print(y_train[image_index]) # The label is 8
    print(x_train.shape)
    plt.imshow(x_train[image_index], cmap='Greys')
    x_train.shape
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])
    
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs=10)
    plt.imshow(img.reshape(28, 28),cmap='Greys')
    pred = model.predict(img.reshape(1,28, 28,1))
    print(pred.argmax())
    return pred.argmax() 

if __name__ == '__main__':
	main()
