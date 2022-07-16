from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop, Adam, SGD, Adagrad
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

classes = ["jiro", "other"]
num_classes = len(classes)


def load_data():
    x_train, x_test, y_train, y_test = np.load(
        "./ramen.npy", allow_pickle=True)
    x_train = x_train.astype("float") / 255
    x_test = x_test.astype("float") / 255
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def train(x_train, y_train, x_test, y_test):
    model = Sequential()

    # Conv Layer 1
    model.add(Conv2D(64, (3, 3), padding='same',
              input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    # Pooling Layer 1
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    # Conv Layer 2
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # Pooling Layer 2
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Conv Layer 3
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(96))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Output Layer
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # Optimizer
    rms = RMSprop(lr=1e-4, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=rms, metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
                     batch_size=128,
                     epochs=30,
                     verbose=0,
                     validation_data=(x_test, y_test))
    return model, hist


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    model, hist = train(x_train, y_train, x_test, y_test)
    model.save('./cnn.h5')

    plt.subplot(1, 2, 1)
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc="upper left")
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc="upper left")
    plt.tight_layout()
    plt.show()

    score = model.evaluate(x_test, y_test, verbose=1)
    print("accuracy=", score[1], "loss=", score[0])
