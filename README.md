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
С помощью python3.8 разработать нейронную сеть прямого распростронения, обычную и свёрточную нейронные сети для классификации картинок. Создать архетектуру нейронной сети и обучить её. Классифицировать картинки в соответствии с вариантом.

Выполнение работы
-----------------
Нейронная сеть прямого распространения - это искусственная нейронная сеть,в которой соединения между узлами не образуют цикла. Мы подаём на вход "сырые" картинки, а на выходе получаем какую-то вероятность пренадлежности какому-то классу. FFNN(Feedforward Neural Network) данные сразу передаются в нейронную сеть.

<p align="center">
  <img src="https://vitalflux.com/wp-content/uploads/2020/10/feed_forward_neural_network-1.gif" />
</p>

Свёрточная нейронная сеть, это такая же нейронная сеть, но прежде чем подать данные их предвательно обрабатывают для получения вектора фитч. CNN(Convolutional Neural Network)-предварительная обработка данных.

<p align="center">
  <img src="https://miro.medium.com/max/1000/1*BIpRgx5FsEMhr1k2EqBKFg.gif" />
</p>

Детекция – задача, в рамках которой необходимо выделитьнесколько объектов на изображении посредством нахождения координат их ограничивающих рамок и классификации этих ограничивающих рамок из множества заранее известных классов.
* Сегментация - поиск групп пикселей, каждая из которых обладает определенными признаками.
Сверточные слои обычно рисуются в виде набора плоскостей или объемов. Каждая плоскость на таком рисунке или каждый срез в этом объеме — это, по сути, один нейрон, который реализует операцию свертки. По сути, это матричный фильтр,
который трансформирует исходное изображение в какое-то другое, и это можно делать много раз.
* Слои субдискретизации (Subsampling или Pooling) просто уменьшают размер изображения: было 200х200 ps, послеSubsampling стало 100х100 ps. По сути, усреднение.
* Полносвязные слои обычно персептрон использует для классификации.

Результат выполнения программы

Нейронная сеть прямого распространения
--------------------------------------


![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/FFNN_model.png)


Оценка полноты, точности и аккуратности


![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/FFNN_accuracy.png)

График потерь и точности для каждой модели


![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/loss-accuracy-ffnn__epoch_20_.png)

![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/loss-accuracy-ffnn.png)



Свёрточная нейронная сеть
--------------------------



![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/CNN_model.png)


Оценка полноты, точности и аккуратности


![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/CNN_accuracy.png)


График потерь и точности для каждой модели


![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/loss-accuracy-сnn.png)

![Gitlab logo](https://bmstu.codes/MorozoFF/lr-6-opc/-/raw/master/loss-accuracy-сnn__epoch_25_.png)

Данные полученные в результате выполнения лабораторной с нестандартными картинками (Ссылка на Google диск с данными)
https://drive.google.com/drive/folders/1bCQIDeO7y4lV-IXUu9NI_t2EC_Bhh_hF?usp=sharing
