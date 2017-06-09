import json
import numpy as np
from numpy import random
from keras.models import Sequential
from keras_diagram import ascii
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, TimeDistributed

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


def x_y_train(u_c_id_train, K=5):
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
                    # L = len(json.loads(line).values()[0])
                    # Y = X_Y[1:L]
                    # X = X_Y[0:L - 1]
                    X = X_Y[-K - 1:-1]
                    # Y = X_Y[-K:]
                    Y = X_Y[-1]
                    yield (np.array([sum(X, [])]), np.array([Y]))


def x_y_test(u_c_id_test, K=5):
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
                    # L = len(json.loads(line).values()[0])
                    # Y = X_Y[1:L]
                    # X = X_Y[0:L - 1]
                    X = X_Y[-K - 1:-1]
                    # Y = X_Y[-K:]
                    Y = X_Y[-1]
                    yield (np.array([sum(X, [])]), np.array([Y]))


def get_basket_form_uid(uid, K=5):
    with open('tafeng_json_data_more_transactions_item_id') as f:
        for line in f:
            if json.loads(line).keys()[0] in uid:
                X_Y = list()
                for i in json.loads(line).values()[0]:
                    list_of_zeros = [0.0] * len(unique_item_id)
                    for j in i:
                        list_of_zeros[j] = 1.0
                    X_Y.append(list_of_zeros)
                # L = len(json.loads(line).values()[0])
                # Y = X_Y[1:L]
                # X = X_Y[0:L - 1]
                # X = X_Y[-K - 1:-1]
                # Y = X_Y[-K:]
                Y = X_Y[-1]
                return np.array(Y)


def build_model(input_dim, hidden_units=[200], dropout=0.5, output_activation='sigmoid'):
    model = Sequential()
    K = 5
    model.add(Dense(hidden_units[0], input_dim=K*input_dim, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim=input_dim, activation=output_activation))
    model.add(Dropout(dropout))
    return model

model = build_model(input_dim=len(unique_item_id))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
# model.fit_generator(x_y_train(0.8, K=5), samples_per_epoch=2560, nb_epoch=5)
uid_train, uid_test = u_c_id_for_train_test(0.8)
print(len(uid_train))
print(len(uid_test))
# model.fit_generator(x_y_train(uid_train, K=5), samples_per_epoch=256, nb_epoch=1)
model.fit_generator(x_y_train(uid_train, K=5), samples_per_epoch=len(uid_train), nb_epoch=1)
# scores = model.evaluate_generator(x_y_train(uid_train, K=5), 10)
scores = model.evaluate_generator(x_y_train(uid_train, K=5), len(uid_train))
print(scores)
print("Accuracy: %.2f%%" % (scores[1]*100))
# model.fit(X, Y, nb_epoch=3, batch_size=64)
# model.fit(example_input, example_output, batch_size=16)
# prediction = model.predict_generator(x_y_test(uid_test, K=5), val_samples=7555)
prediction = model.predict_generator(x_y_test(uid_test, K=5), val_samples=len(uid_test))
print(prediction[0])
# print(prediction[0][4])# K-1
print(get_basket_form_uid(uid_test[0], K=5)[1])
print(len(prediction))
print([i for i,v in enumerate(prediction[0]) if v > 0.5])
print([i for i,v in enumerate(get_basket_form_uid(uid_test[0], K=5)) if v > 0.5])
print(uid_test[0])
# print(ascii(model))













# embedding_vector_length = 100
# model = Sequential()
# model.add(Embedding(input_dim=dim, output_dim=embedding_vector_length, input_length=1, dropout=0.2))
# model.add(LSTM(output_dim=64, return_sequences=True, forget_bias_init='one'))
# model.add(Dense(output_dim=dim))
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.fit(X, Y, nb_epoch=3, batch_size=64)
# model = Sequential()
# model.add(LSTM(32, return_sequences=True,
#                input_shape=(L-1, dim)))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(output_shape=(L-1, dim)))  # return a single vector of dimension 32
# # model.add(Dense(output_size=(L-1, dim), activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.fit(X, Y, nb_epoch=3, batch_size=64)
# max_len = 100
# X = pad_sequences(X, maxlen=dim)
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))
# print(X.shape)
# print(Y.shape)
# Y = pad_sequences(Y, maxlen=max_len)
# Y = np.reshape(Y, (Y.shape[0], max_len, 1))
# model = Sequential()
# model.add(LSTM(128, input_shape=(X.shape[0], X.shape[1])))
# model.add(Dense(Y.shape[1], activation='softmax'))
# model.add(LSTM(128, input_shape=(X.shape[1], 1)))
# model.add(Dense(Y.shape[1], activation='softmax'))
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.fit(X, Y, nb_epoch=3, batch_size=64)

# model = Sequential()
# model.add(LSTM(32, return_sequences=True,
#                input_shape=(, dim)))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(output_shape=(L-1, dim)))  # return a single vector of dimension 32
# # model.add(Dense(output_size=(L-1, dim), activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.fit(X, Y, nb_epoch=3, batch_size=64)