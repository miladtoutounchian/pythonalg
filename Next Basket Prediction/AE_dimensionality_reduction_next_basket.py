import json
import numpy as np
from numpy import random
import math
from keras.models import Sequential
from keras_diagram import ascii
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, TimeDistributed
from keras.models import load_model
import cardinality
import cPickle as pickle

data = list()
with open('tafeng_json_data_more_transactions') as f:
    for line in f:
        data.append(sum(list(json.loads(line).values()[0]), []))

all_item_id = sum(data, [])

print(len(list(set(all_item_id))))
# print()

unique_item_id = list(set(all_item_id))


def get_unique_customer_id(filename):
    all_customer_id = list()
    with open(filename) as f:
        for line in f:
            all_customer_id.append(json.loads(line).keys()[0])
    unique_customer_id = list(set(all_customer_id))
    return unique_customer_id


def u_c_id_for_train_test(ratio):
    u_c_id = get_unique_customer_id('tafeng_json_data_more_transactions_item_id')
    random.shuffle(u_c_id)
    u_c_id_train = u_c_id[:int(ratio * len(u_c_id))]
    u_c_id_test = list(set(u_c_id)-set(u_c_id_train))
    return u_c_id_train, u_c_id_test


def x_y_train_ae(u_c_id_train):
    with open('tafeng_json_data_more_transactions_item_id') as f:
        for line in f:
            if json.loads(line).keys()[0] in u_c_id_train:
                X_Y = list()
                for i in json.loads(line).values()[0]:
                    list_of_zeros = [0.0] * len(unique_item_id)
                    for j in i:
                        list_of_zeros[j] = 1.0
                    X_Y.append(list_of_zeros)
                L = len(json.loads(line).values()[0])
                for j in range(L):
                    X = X_Y[j]
                    yield (np.array([X]), np.array([X]))

uid_train, uid_test = u_c_id_for_train_test(1)
print(len(uid_train))
print(len(uid_test))

print(cardinality.count(x_y_train_ae(uid_train)))

ae = Sequential()
inputLayer = Dense(100, input_shape=(len(unique_item_id),), activation='tanh')
ae.add(inputLayer)
output = Dense(len(unique_item_id), activation='sigmoid')
ae.add(output)
ae.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
ae.fit_generator(x_y_train_ae(uid_train), samples_per_epoch=cardinality.count(x_y_train_ae(uid_train)), nb_epoch=200)

w1 = ae.layers[0].get_weights()[0]
b1 = ae.layers[0].get_weights()[1]
w2 = ae.layers[1].get_weights()[0]
b2 = ae.layers[1].get_weights()[1]
pickle.dump(w1, open("w1.p", "wb"))
pickle.dump(b1, open("b1.p", "wb"))
pickle.dump(w2, open("w2.p", "wb"))
pickle.dump(b2, open("b2.p", "wb"))


