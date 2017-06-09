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
import cPickle as pickle


w1 = pickle.load(open("w1.p", "rb"))
b1 = pickle.load(open("b1.p", "rb"))
w2 = pickle.load(open("w2.p", "rb"))
b2 = pickle.load(open("b2.p", "rb"))


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


def x_y_train_rnn(u_c_id_train):
    while 1:
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
                    X = X_Y[0:L - 1]
                    X_transformed_ld = np.array([np.tanh(np.dot(x, w1) + b1) for x in X])
                    Y = X_Y[1:L]
                    Y_transformed_ld = np.array([np.tanh(np.dot(y, w1) + b1) for y in Y])
                    # X = X_Y[-K - 1:-1]
                    # Y = X_Y[-K:]
                    # yield (np.array([X]), np.array([Y]))
                    yield np.array([X_transformed_ld]), np.array([Y_transformed_ld])


def x_y_test_rnn(u_c_id_test):
    while 1:
        with open('tafeng_json_data_more_transactions_item_id') as f:
            for line in f:
                if json.loads(line).keys()[0] in u_c_id_test:
                    X_Y = list()
                    for i in json.loads(line).values()[0]:
                        list_of_zeros = [0.0] * len(unique_item_id)
                        for j in i:
                            list_of_zeros[j] = 1.0
                        X_Y.append(list_of_zeros)
                    L = len(json.loads(line).values()[0])
                    X = X_Y[0:L - 1]
                    X_transformed_ld = np.array([np.dot(x, w1) + b1 for x in X])
                    Y = X_Y[1:L]
                    # Y_transformed_ld = np.array([np.dot(y, w1) + b1 for y in Y])
                    # X = X_Y[-K - 1:-1]
                    # Y = X_Y[-K:]
                    yield np.array([X_transformed_ld]), np.array([Y]), json.loads(line).values()[0]

uid_train, uid_test = u_c_id_for_train_test(0.8)


def build_model(input_dim, input_length=None, hidden_units=[150, 150], dropout=0.1, output_activation='sigmoid'):
    model = Sequential()
    model.add(LSTM(hidden_units[0], input_shape=(input_length, input_dim), return_sequences=True))
    model.add(Dropout(dropout))
    for h in hidden_units[1:]:
        model.add(LSTM(h, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(TimeDistributed(Dense(input_dim, activation=output_activation)))
    return model


model = build_model(input_dim=100)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# F = x_y_train_rnn(uid_train)
model.fit_generator(x_y_train_rnn(uid_train), samples_per_epoch=len(uid_train), nb_epoch=200)
# scores = model.evaluate_generator(x_y_train(uid_train, K=5), 10)
# scores = model.evaluate_generator(x_y_train(uid_train), len(uid_train))
# print(scores)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# model.fit(X, Y, nb_epoch=3, batch_size=64)
# model.fit(example_input, example_output, batch_size=16)
# prediction = model.predict_generator(x_y_test(uid_test, K=5), val_samples=7555)
# prediction = model.predict_generator(x_y_test(uid_test), val_samples=len(uid_test))
# G = x_y_test(uid_test)
# print(len(uid_train))
# print(uid_train)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

vfunc = np.vectorize(sigmoid)

G = x_y_test_rnn(uid_train)
s = 0
for i in range(1):
    X_te, Y_te, b = next(G)
    prediction = model.predict(X_te)
    prediction_transformed_hd = np.array([vfunc(np.dot(x, w2) + b2) for x in prediction])
    # Y_te_transformed_hd = np.array([vfunc(np.dot(x, w2) + b2) for x in Y_te])
    # prediction_transformed_hd = np.array([vfunc(np.dot(w2, x) + b2) for x in prediction])
    # Y_te_transformed_hd = np.array([vfunc(np.dot(w2, x) + b2) for x in Y_te])
    # print(len(prediction[0]))
    # print(len(Y_te[0]))
    # print(prediction[0])
    # print(Y_te[0])
    for j in range(len(prediction_transformed_hd[0])):
        print([i for i, v in enumerate(prediction_transformed_hd[0][j]) if v > 0.25])
        print(max(prediction_transformed_hd[0][j]))
        print([i for i, v in enumerate(Y_te[0][j]) if v > 0.25])
        # print(b[j+1])
        # print(np.nonzero(Y_train[0][j]))
        s += np.linalg.norm(Y_te[0][j] - prediction_transformed_hd[0][j])
print(s)
