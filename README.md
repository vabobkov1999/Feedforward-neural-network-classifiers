# Лабораторная работа №6

Классификаторы на основе нейронных сетей прямого распространения
----------------------------------------------------------------


| 🔢  | Ход работы   | ℹ️ |
| ------------- | ------------- |------------- |
| 1️⃣  |  Установить библиотеку keras, tensorflow. | ✅ |
| 2️⃣ | Скачать датасет с симпсонами варианту (для каждого класса отдельная директория).  |✅  |
| 3️⃣ | Создать архитектуру нейронной сети прямого распространения с двумя полносвязными скрытыми слоями и сверточной нейронной сети с двумя сверточными и двумя полносвязными слоями и скомпилировать модели (model.summary()). |✅  |



Цель работы
------------
С помощью python3.8 разработать нейронные сети с прямой связью (обычную и свёрточную нейронные сети) для классификации картинок. Создать архетектуру нейронной сети и обучить её. Классифицировать картинки в соответствии с вариантом.

Выполнение работы
-----------------

#### Feedforward Neural Network

Нейронная сеть прямого распространения - это искусственная нейронная сеть, в которой соединения между узлами не образуют цикла. Мы подаём на вход "сырые" картинки, а на выходе получаем какую-то вероятность принадлежности какому-то классу. FFNN(Feedforward Neural Network) данные сразу передаются в нейронную сеть.

<p align="center">
  <img src="https://vitalflux.com/wp-content/uploads/2020/10/feed_forward_neural_network-1.gif" />
</p>

В начале программы с FFNN сразу после библиотек идут пути к папкам с картинками.
Так путь train_path ведёт к тренировочным папкам по которым будет учиться нейронная сети. Путь test_path ведёт  к файлам, которые мы будем классифицировать. Путь result_path указывает куда будут сохраняются файлы которые нейронная сеть обработала. А путь plot_path указывает  куда сохранить график.

```python
train_path = "dataset/train"
test_path = "dataset/test"
result_path = "dataset/saved_ffnn"
plot_path = "../../Desktop/lr-6-opc-master/loss-accuracy-ffnn.png"
```

Далее происходит инициализация данных  и меток. Так массив data будет содержать набор данных о картинках т.е. вектора связанные с картинкой, которые представляют RGB матрицу (Матрица размерности = ширина х высоту х (3(так как 3 цвета)). А массив labels содержит название картинок.

```python
data = []
labels = []
```

С помощью метода listdir смотрим какие папки находятся на пути.

```python
train_labels = os.listdir(train_path)
```

С помощью метода sort() сортируем по имени.

```python
train_labels.sort()
```

Цикл по изображениям из папки train - вытаскиваем картинки и кладем в data, их класс (имя симпсона) в labels.
С помощью метода path.join запоминанием пути к каждой папке, current_label название текущей папки. Далее создаём пути для каждой папки path.

```python
dir = os.path.join(train_path, training_name)
  current_label = training_name
```

Далее создаём пути для тренировки.
```python
path = train_path + "/" + training_name
```
С помощью метода listdir  определяем содержание папки.
```python
x = os.listdir(dir)
```
Пробегаем по каждому файлу и создаём путь для каждой картинки.
С помощью метода imread считываем картинку.
С помощью метода resize преобразуем изображение в размер 32х32, а с помощью метода
flatten(его используют так как нейронка работает только с одномерными массивами ) преобразуем картинку в одномерный массив, где его элементами будут пиксели RGB  расцветки

```python
for y in x:
      file = path + "/" + str(y)

      image = cv2.imread(file)
      image = cv2.resize(image, (32, 32)).flatten()
      data.append(image)

      labels.append(current_label)
```

Следующим шагом разбиваем данные на обучающую и тестовую выборки, при этом используя 80% данных для обучения(trainX, testX ) и оставшиеся 20% (trainY, testY) для тестирования. Для разбиения используем метод train_test_split. Фрагмент кода с этим методом представлен ниже.

```python
seed = 15 # random.randint(1, 100)
print("seed =", seed)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=seed)
print(testY)
```
Параметр random_state=seed влияет на принцип деления данных по закономерности
конвертирует метки из целых чисел в векторы (seed  указывается рандомное значение влияет на принцип деления данных)

Используя метод LabelBinarizer для каждого класса формируем вектор(к примеру если классов 5, то вектор длинны 5)

```python
lb = LabelBinarizer()
```

Теперь перейдем к основному этапу программы, а именно создадим архитектуру нейронной сети с помощью модели Sequential. С помощью метода add добавляем слои( Dense спрятанный слой).
input_shape входная размерность указывается только в первом слое или методе и  равна 32х32х37
Далее определим выходное значение нейрона в зависимости от результата взвешенной суммы входов и порогового значения для этого используем функцию активации activation="sigmoid".
Добавим ещё один слой 512 точек используем функцию активации activation="sigmoid".
Последний слой имеет длину lb.classes_ и функцию активации activation="softmax.
И наконец инициализируем скорость обучения и общее число эпох EPOCHS = 20 инициализация эпох

```python
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

EPOCHS = 20

```

Скомпилируем модель, используя метод Adam в качестве оптимизатора для более быстрых вычислений и категориальную кросс-энтропию в качестве функции потерь.
Используем метод compile, функцию потерь , кросс-энтропию и метрику точности компилируем модель нейронной сети.
Используем метод summary построим таблицу архитектуры нейронной сети.

```python
print("[INFO] training network...")
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# архитектура нейронки
model.summary()
```

C помощью метода fit загружаем в модель тренировочные и тестовые данные и количество эпох
(параметр batch_size управляет размером пакетов в компьютере).

```python
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)
```

Произведём оценку нейронной сети. Для этого воспользуемся методом predict, который предсказывает вероятности принадлежности классам и построим табличку.

```python
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

```

В завершении построим графики потерь и точности
loss и accuracy значения для тренировочных данных.
val_loss и val_accuracy значения для тестовых данных.

Строим сетку графика
Параметр N = np.arange на сколько он будет длинным по оси.

```python
N = np.arange(0, EPOCHS)
```

Метод plot добавляет график.
В h содержится матрица параметров для каждой эпохи, а с помощью метода history выдает значения для каждой эпохи.

```python
plt.plot(N, H.history["loss"], label="training loss")
plt.plot(N, H.history["val_loss"], label="validation loss")
plt.plot(N, H.history["accuracy"], label="training accuracy")
plt.plot(N, H.history["val_accuracy"], label="validation accuracy")
```

Начинаем предсказывать принадлежность изображения к классу
 С помощью метода listdir запоминанием название картинок и путь к ним, а также считываем его.

 ```python
test_labels = os.listdir(test_path)
 ```


Загружаем входное изображение и меняем его размер на необходимый
 Для каждой картинки мы запоминаем путь, считываем и копируем его.

 ```python
 path_for_save = result_path + "/" + test_file
   test_file = test_path + "/" + test_file
   image = cv2.imread(test_file)
   image_copy = image.copy()
   image = cv2.resize(image, (32, 32)).flatten()
 ```


Выводим название класса и вероятность принадлежности на картинку.

 ```python
text = f'{label}: {round(predictions[0][i] * 100, 2)}%'
 ```


#### Convolutional Neural Network

Свёрточная нейронная сеть - это такая же нейронная сеть, но прежде чем подать данные их предварительно обрабатывают для получения вектора фитч. CNN(Convolutional Neural Network)-предварительная обработка данных.

<p align="center">
  <img src="https://miro.medium.com/max/1000/1*BIpRgx5FsEMhr1k2EqBKFg.gif" />
</p>

При обработки изображений CNN выполняются следующие операции:
* Классификация - это первый этап обработки изображения на, котором происходит разделение объектов на определённое количество классов по определённым признакам, при этом задано конечное множество объектов, для которых уже определен класс. Для остальных необходимо построить алгоритм, способный классифицировать их. Так же для классификации используются полносвязные слои.

<p align="center">
  <img src="https://btechmag.com/wp-content/uploads/2020/11/UNSupervised-Learning.gif" />
</p>

* Детекция – задача, в рамках которой необходимо выделить несколько объектов на изображении посредством нахождения координат их ограничивающих рамок и классификации этих ограничивающих рамок из множества заранее известных классов.

<p align="center">
  <img src="http://download.smartlife.global/files/face-recog-4.gif" />
</p>

* Сегментация - поиск групп пикселей, каждая из которых обладает определенными признаками.
Сверточные слои обычно рисуются в виде набора плоскостей или объемов. Каждая плоскость на таком рисунке или каждый срез в этом объеме — это, по сути, один нейрон, который реализует операцию свертки. По сути, это матричный фильтр,
который трансформирует исходное изображение в какое-то другое, и это можно делать много раз.

<p align="center">
  <img src="https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/opencv-mask-rcnn/mask_rcnn_example03.gif" />
</p>

* Слои субдискретизации (Subsampling или Pooling) просто уменьшают размер изображения: было 200х200 ps, после Subsampling стало 32х32 ps. По сути, усреднение.

<p align="center">
  <img src="https://miro.medium.com/max/1400/1*kOThnLR8Fge_AJcHrkR3dg.gif" />
</p>


### Результат выполнения программы

#### Нейронная сеть прямого распространения



![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/FFNN_model.png)


Оценка полноты, точности и аккуратности


![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/FFNN_accuracy.png)

График потерь и точности для каждой модели


![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/loss-accuracy-ffnn__epoch_20_.png)

![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/loss-accuracy-ffnn.png)



#### Свёрточная нейронная сеть




![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/CNN_model.png)


Оценка полноты, точности и аккуратности


![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/CNN_accuracy.png)


График потерь и точности для каждой модели


![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/loss-accuracy-сnn.png)

![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/loss-accuracy-сnn__epoch_25_.png)

Данные полученные в результате выполнения ЛР с нестандартными картинками [Ссылка на Google диск с данными](https://drive.google.com/drive/folders/1bCQIDeO7y4lV-IXUu9NI_t2EC_Bhh_hF?usp=sharing)

Так же напоминаю для тех кому интересно выполнить задание самому или протестировать данную программу, то прошу перейти [сюда](https://drive.google.com/drive/folders/1b_molbj8z6JhHV6r178AeI1XpQezehsm?usp=sharing "Практикум по машинному обучению")
