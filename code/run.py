import pandas as pd
import os
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf

def is_none(num):
    if num is None:
        return 0.0
    else:
        return num

def add_output(output, hor_name):
    for i in range(59, 1225, 1):
        one_output = []
        one_output.append(i)
        one_output.append(is_none(l2_horizons_train[hor_name][i]))
        output.append(one_output)

def add_train(train, hor_name, start, stop):
    for i in range(start, stop, 1):
        one_train_set = []
        one_train_set.append(i)
        one_train_set.append(is_none(l2_horizons_train[hor_name][i]))
        one_train_set.append(is_none(all_data_l2[i+1][int(l2_horizons_train[hor_name][i])]))
        one_train_set.append(is_none(all_data_l2[i+1][int(l2_horizons_train[hor_name][i+1])]))
        one_train_set.append(is_none(all_data_l2[i+1][int(l2_horizons_train[hor_name][i-1])]))
        train.append(one_train_set)
    print(train[0])

ROOT = "D:/DEV/Pythom/Hackaton/"
DATA_DIR = ROOT + "data"
l1_nrows = 1512
l2_nrows = 1248
n_horizons = 4
y_height = 3001

# Загружаем значения высот в узлах сетки для срезов L1 и L2

all_data_l1 = np.load(os.path.join(DATA_DIR, "all_data_L1.npy"))
all_data_l2 = np.load(os.path.join(DATA_DIR, "all_data_L2.npy"))

assert all_data_l1.shape == (l1_nrows, y_height), "Неправильный размер all_data_L1.npy"
assert all_data_l2.shape == (l2_nrows, y_height), "Неправильный размер all_data_L2.npy"

# Загружаем горизонты

l1_horizons_train = pd.read_csv(os.path.join(DATA_DIR, "L1_horizons_train.csv"))
l2_horizons_train = pd.read_csv(os.path.join(DATA_DIR, "L2_horizons_train.csv"))

assert l1_horizons_train.shape == (l1_nrows, n_horizons+1)
assert l2_horizons_train.shape == (l2_nrows, n_horizons+1)

# Нам необходимо предсказать значения второго горизонта среза L1 для всех x в промежутке 522, ..., 1451

train = []
add_train(train, 'hor_1', 58, 1224)
add_train(train, 'hor_3', 58, 1224)
add_train(train, 'hor_4', 58, 1224)
print(len(train))

output = []
add_output(output, 'hor_1')
add_output(output, 'hor_3')
add_output(output, 'hor_4')

test = []

for i in range(58, 1224, 1):
    one_train_set = []
    one_train_set.append(i)
    one_train_set.append(is_none(l2_horizons_train['hor_2'][i]))
    one_train_set.append(is_none(all_data_l2[i+1][int(l2_horizons_train['hor_2'][i])]))
    one_train_set.append(is_none(all_data_l2[i+1][int(l2_horizons_train['hor_2'][i+1])]))
    one_train_set.append(is_none(all_data_l2[i+1][int(l2_horizons_train['hor_2'][i-1])]))
    test.append(one_train_set)

test_output = []
for i in range(59, 1225, 1):
    one_output = []
    one_output.append(i)
    one_output.append(is_none(l2_horizons_train['hor_2'][i]))
    test_output.append(one_output)

pred = []
add_train(pred, 'hor_2', 58, 521)

test_output = np.array(test_output)
test = np.array(test)
output = np.array(output)
train = np.array(train)

test_output /= 1000000000
test /= 1000000000
output /= 1000000000
train /= 1000000000

output = to_categorical(output)
test_output = to_categorical(test_output)

model = Sequential()
model.add(Dense(5, input_dim=5, activation='sigmoid'))
model.add(Dense(5, input_dim=5, activation='sigmoid'))
model.add(Dense(2))

optimizer = tf.keras.optimizers.RMSprop(0.0001)

model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])

history = model.fit(train, output,
            epochs=50, 
            verbose=2, 
            validation_data=(test, test_output))

test_predictions = model.predict(pred)
test_predictions *= 1000

with open(os.path.join(DATA_DIR, "sample_submission.csv"), 'r') as old:
    new_sample = open(os.path.join(DATA_DIR, "my_submission.csv"), 'w')
    new_sample.write('x,y\n')
    old.readline()
    for i in test_predictions:
        x = old.readline().split(',')[0]
        new_sample.write('{},{:.2f}\n'.format(x, i[1]))
    for i in range(467):
        new_sample.write('{},{:.2f}\n'.format(984+i, 965.6))


