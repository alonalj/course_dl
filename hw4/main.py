import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.layers import CuDNNLSTM, Bidirectional, Dropout, TimeDistributed, BatchNormalization, Input
from keras.utils import to_categorical
from keras.models import Model
from keras.callbacks import ModelCheckpoint


# load the dataset but only keep the top words, zero the rest. Introduce special tokens.
VOCABULARY_SIZE = 20000
MAX_SAMPLES = 40000
MAX_LEN = 100

(sentences, sentiment), _ = imdb.load_data(num_words=VOCABULARY_SIZE)

word_to_id = imdb.get_word_index()
word_to_id = {k: (v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<OOV>"] = 2
word_to_id["<EOS>"] = 3
id_to_word = {v: k for k, v in word_to_id.items()}


def sample(preds, temperature=1.0):
    """Helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def word_level_time_dist():
    LSTM_SIZE = 256

    X_train = sequence.pad_sequences(sentences, maxlen=MAX_LEN, truncating='post')
    y_train = np.roll(X_train, -1, axis=-1)
    y_train = y_train[:, :, np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_train), np.array(y_train), test_size=0.1)

    input = Input(shape=(MAX_LEN, ))
    se1 = Embedding(VOCABULARY_SIZE, 200, input_length=MAX_LEN)(input) #, mask_zero=True
    se2 = Dropout(0.3)(se1)

    # Stack 1
    lstm1 = LSTM(LSTM_SIZE, return_sequences=True)(se2)
    bn1 = BatchNormalization()(lstm1)
    do1 = Dropout(0.3)(bn1)

    # Stack 2
    lstm2 = LSTM(LSTM_SIZE, return_sequences=True)(do1)
    bn2 = BatchNormalization()(lstm2)
    do2 = Dropout(0.3)(bn2)

    # Stack 3
    lstm3 = LSTM(LSTM_SIZE, return_sequences=True)(do2)
    bn3 = BatchNormalization()(lstm3)
    do3 = Dropout(0.3)(bn3)

    fc1 = TimeDistributed(Dense(512, activation='relu'))(do4)
    bn5 = BatchNormalization()(fc1)
    do5 = Dropout(0.3)(bn5)
    output = TimeDistributed(Dense(VOCABULARY_SIZE, activation='softmax', input_shape=(MAX_LEN, LSTM_SIZE)))(do5)

    model = Model(inputs=input, outputs=output)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam')
    print(model.summary())

    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-timedist-3lstm256-1fc512-vocab20000-maxlen100'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=20, batch_size=128, callbacks=[checkpoint])

    seed = []
    seed.append(word_to_id['<START>'])
    seed.append(word_to_id['the'])
    seed.append(word_to_id['movie'])
    seed.append(word_to_id['was'])

    for i in range(3, MAX_LEN):
        model.reset_states()
        seed_padded = sequence.pad_sequences([seed], maxlen=MAX_LEN, truncating='post', padding='post')
        y = model.predict(seed_padded)[0][i]
        next_word_id = sample(y, temperature=0.7)
        seed.append(next_word_id)

    for i in range(MAX_LEN):
        print("{} ".format(id_to_word[seed[i]]))



def word_level():
    X_train = sequence.pad_sequences(sentences, maxlen=MAX_LEN, truncating='post')
    y_train = np.roll(X_train, -1, axis=-1)
    y_train[:, -1] = word_to_id["<EOS>"]
    # y_train = y_train[:, :, np.newaxis]

    X_train_ext = []
    y_train_ext = []

    # For each sentence
    for i, x in enumerate(X_train):
        # For each word in the sentence
        for j in range(2, len(x)):
            if len(X_train_ext) >= MAX_SAMPLES:
                break

            X_train_ext.append(sequence.pad_sequences([x[:j]], maxlen=MAX_LEN, truncating='post', padding='post')[0])
            y_train_ext.append(to_categorical([x[j]], num_classes=VOCABULARY_SIZE)[0])

    X_train = np.array(X_train_ext)
    y_train = np.array(y_train_ext)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

    print(X_train.shape)
    print(y_train.shape)

    LSTM_SIZE = 256

    input = Input(shape=(MAX_LEN,))
    se1 = Embedding(VOCABULARY_SIZE, 100, input_length=MAX_LEN, mask_zero=True)(input)  #
    se2 = Dropout(0.3)(se1)

    # Stack 1
    lstm1 = LSTM(LSTM_SIZE, return_sequences=True)(se2)
    bn1 = BatchNormalization()(lstm1)
    do1 = Dropout(0.3)(bn1)

    # Stack 2
    lstm2 = LSTM(LSTM_SIZE, return_sequences=True)(do1)
    bn2 = BatchNormalization()(lstm2)
    do2 = Dropout(0.3)(bn2)

    # Stack 3
    lstm3 = LSTM(LSTM_SIZE)(do2)
    bn3 = BatchNormalization()(lstm3)
    do3 = Dropout(0.3)(bn3)

    output = Dense(VOCABULARY_SIZE, activation='softmax')(do3)

    model = Model(inputs=input, outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=5, batch_size=128)

    seed = []
    seed.append(word_to_id['<START>'])
    seed.append(word_to_id['the'])
    seed.append(word_to_id['movie'])
    seed.append(word_to_id['was'])

    for i in range(MAX_LEN - 4):
        seed_padded = sequence.pad_sequences([seed], maxlen=MAX_LEN, truncating='post', padding='post')
        result = model.predict(seed_padded)
        id = np.argmax(result)
        seed.append(id)

    for i in range(MAX_LEN):
        print("{} ".format(id_to_word[seed[i]]))