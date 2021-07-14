from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

train_path = "dataset/train"
test_path = "dataset/test"
result_path = "dataset/saved_ffnn"
plot_path = "../../Desktop/lr-6-opc-master/loss-accuracy-ffnn.png"

# инициализируем данные и метки
print("[INFO] loading images...")
data = []
labels = []
train_labels = os.listdir(train_path)
train_labels.sort()
print(train_labels)

# цикл по изображениям из папки train - вытаскиваем картинки и кладем в data, их класс (имя симпсона) в labels
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name

    path = train_path + "/" + training_name
    x = os.listdir(dir)

    for y in x:
        file = path + "/" + str(y)

        image = cv2.imread(file)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)

        labels.append(current_label)

    print("[INFO] processed folder: {}".format(current_label))

print("[INFO] completed loading images...")
print(len(data))
print(len(labels))

# масштабируем интенсивности пикселей в диапазон [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# разбиваем данные на обучающую и тестовую выборки, используя 80% данных для обучения и оставшиеся 20% для тестирования
seed = 15 # random.randint(1, 100)
print("seed =", seed)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=seed)
print(testY)
# seed=15 epochs=20

# конвертируем метки из целых чисел в векторы
# [1, 0, 0] # относится к кошкам
# [0, 1, 0] # относится к собакам
# [0, 0, 1] # относится к панде
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# определим архитектуру 3072-1024-512-3 с помощью Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# инициализируем скорость обучения и общее число эпох
EPOCHS = 20

# компилируем модель, используя Adam как оптимизатор и категориальную кросс-энтропию в качестве функции потерь
print("[INFO] training network...")
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# архитектура нейронки
model.summary()

# обучаем нейросеть
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

# оцениваем нейросеть
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# строим графики потерь и точности
plt.style.use("ggplot")
N = np.arange(0, EPOCHS)
plt.figure()
plt.plot(N, H.history["loss"], label="training loss")
plt.plot(N, H.history["val_loss"], label="validation loss")
plt.plot(N, H.history["accuracy"], label="training accuracy")
plt.plot(N, H.history["val_accuracy"], label="validation accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(plot_path)
plt.show()

# начинаем предсказывать принадлежность к классу
print("[INFO] loading images for testing...")
test_labels = os.listdir(test_path)
test_labels.sort()
print("test labels =", test_labels)

# цикл по изображениям test
for test_file in test_labels:
    # загружаем входное изображение и меняем его размер на необходимый
    path_for_save = result_path + "/" + test_file
    test_file = test_path + "/" + test_file
    image = cv2.imread(test_file)
    image_copy = image.copy()
    image = cv2.resize(image, (32, 32)).flatten()
    # масштабируем значения пикселей к диапазону [0, 1]
    image = image.astype("float") / 255.0
    image = image.reshape((1, image.shape[0]))

    # массив предсказания класса картинки
    predictions = model.predict(image)
    # находим индекс класса с наибольшей вероятностью
    i = predictions.argmax(axis=1)[0]
    label = lb.classes_[i]

    # выводим название класса и вероятность принадлежности на картинку
    text = f'{label}: {round(predictions[0][i] * 100, 2)}%'
    cv2.putText(image_copy, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 0, 255), 2)
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite(path_for_save, image_copy)
