import numpy as np
import pdb
import string
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
IMDB_VOCABULARY_SIZE = 20000
CHAR_VOCABULARY_SIZE = 101
MAX_SAMPLES = 40000
MAX_LEN = 100
CHAR_MAX_LEN = 200


def sample_multinomial(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def sample_argmax(preds):
    return np.argmax(preds)


class TextGenModel:
    def __init__(self, type='WL', lstm_size=256, fc_size=256, do_ratio=0.3, embed_size=200,
                 pretrained=None, cl_in_shape=None):
        self._type = type
        self._lstm_size = lstm_size
        self._fc_size = fc_size
        self._do_ratio = do_ratio
        self._embed_size = embed_size
        self._cl_in_shape = cl_in_shape

        if type == 'CL' and cl_in_shape is None:
            raise ValueError('Must enter cl_in_shape when using CL model.')

        self._filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-' + str(self._type).lower() +\
                         '-4lstm' + str(self._lstm_size) + '-1fc' + str(self._fc_size) +\
                         '-vocab' + str(IMDB_VOCABULARY_SIZE) + '-maxlen' + str(MAX_LEN)

        self.model = self._create_model()

        if pretrained is not None:
            self.model.load_weights(pretrained)

    def _create_model(self):
        if self._type == 'WL':
            input = Input(shape=(MAX_LEN,))
            se1 = Embedding(IMDB_VOCABULARY_SIZE, self._embed_size, input_length=MAX_LEN, mask_zero=True)(input)
            se2 = Dropout(self._do_ratio)(se1)

            # Stack 1
            lstm1 = LSTM(self._lstm_size, return_sequences=True)(se2)
            bn1 = BatchNormalization()(lstm1)
            do1 = Dropout(self._do_ratio)(bn1)

            # Stack 2
            lstm2 = LSTM(self._lstm_size, return_sequences=True)(do1)
            bn2 = BatchNormalization()(lstm2)
            do2 = Dropout(self._do_ratio)(bn2)

            # Stack 3
            lstm3 = LSTM(self._lstm_size, return_sequences=True)(do2)
            bn3 = BatchNormalization()(lstm3)
            do3 = Dropout(self._do_ratio)(bn3)

            # Stack 4
            lstm4 = LSTM(self._lstm_size, return_sequences=True)(do3)
            bn4 = BatchNormalization()(lstm4)
            do4 = Dropout(self._do_ratio)(bn4)

            fc1 = TimeDistributed(Dense(self._fc_size, activation='relu'))(do4)
            bn5 = BatchNormalization()(fc1)
            do5 = Dropout(self._do_ratio)(bn5)
            output = TimeDistributed(Dense(IMDB_VOCABULARY_SIZE, activation='softmax',
                                           input_shape=(MAX_LEN, self._lstm_size)))(do5)

            model = Model(inputs=input, outputs=output)
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam')
            print(model.summary())

            return model

        elif self._type == 'CL':
            input = Input(shape=self._cl_in_shape)

            # Stack 1
            lstm1 = LSTM(self._lstm_size, return_sequences=True)(input)
            bn1 = BatchNormalization()(lstm1)
            do1 = Dropout(self._do_ratio)(bn1)

            # Stack 2
            lstm2 = LSTM(self._lstm_size, return_sequences=True)(do1)
            bn2 = BatchNormalization()(lstm2)
            do2 = Dropout(self._do_ratio)(bn2)

            # Stack 3
            lstm3 = LSTM(self._lstm_size, return_sequences=True)(do2)
            bn3 = BatchNormalization()(lstm3)
            do3 = Dropout(self._do_ratio)(bn3)

            # Stack 4
            lstm4 = LSTM(self._lstm_size, return_sequences=True)(do3)
            bn4 = BatchNormalization()(lstm4)
            do4 = Dropout(self._do_ratio)(bn4)

            fc1 = TimeDistributed(Dense(self._fc_size, activation='relu'))(do4)
            bn5 = BatchNormalization()(fc1)
            do5 = Dropout(self._do_ratio)(bn5)
            output = TimeDistributed(Dense(CHAR_VOCABULARY_SIZE, activation='softmax'))(do5)

            model = Model(inputs=input, outputs=output)
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam', metrics=['accuracy'])
            print(model.summary())

            return model

        else:
            raise ValueError('Type can only get WL (word-level) and CL (character-level) values.')




def main():
    pretrained = "/home/gilsho/236606/hw4/model-ep056-loss4.630-val_loss4.944-wl-4lstm256-1fc256-vocab20000-maxlen100"
    pretrained = None

    # Load and preprocess the dataset
    (sentences, sentiment), _ = imdb.load_data(num_words=IMDB_VOCABULARY_SIZE)

    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<OOV>"] = 2
    word_to_id["<EOS>"] = 3
    id_to_word = {v: k for k, v in word_to_id.items()}

    # W-L
    ##X_train = sequence.pad_sequences(sentences, maxlen=MAX_LEN, truncating='post')
    ##y_train = np.roll(X_train, -1, axis=-1)
    ##y_train = y_train[:, :, np.newaxis]


    # C-L
    characters = ['<PAD>', '<START>', '<EOS>'] + list(string.printable)
    characters.remove('\x0b')
    characters.remove('\x0c')
    assert (len(characters) == CHAR_VOCABULARY_SIZE)
    char2ind = {c: i for i, c in enumerate(characters)}

    X_train_words = sequence.pad_sequences(sentences, maxlen=MAX_LEN, truncating='post')
    X_train = []
    for i, s in enumerate(X_train_words):
        X_train.append([])
        for w in s:
            word = id_to_word[w]
            for c in word:
                try:
                    X_train[i].append(char2ind[c])
                except:
                    print("Skipping {} character".format(c))

            X_train[i].append(char2ind[' '])

    X_train = sequence.pad_sequences(X_train, maxlen=CHAR_MAX_LEN, truncating='post')
    y_train = np.roll(X_train, -1, axis=-1)

    X_train = np.array([to_categorical(x, num_classes=CHAR_VOCABULARY_SIZE) for x in X_train])
    y_train = np.array([to_categorical(y, num_classes=CHAR_VOCABULARY_SIZE) for y in y_train])

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_train), np.array(y_train), test_size=0.1)


    # Model and training
    my_model = TextGenModel(type='CL', pretrained=pretrained, cl_in_shape=X_train.shape[1:])

    if pretrained is None:
        checkpoint = ModelCheckpoint(my_model._filepath, monitor='val_loss', verbose=1,
                                     save_best_only=False, mode='min')
        my_model.model.fit(X_train, y_train,
                           validation_data=(X_test, y_test), epochs=30,
                           batch_size=128, callbacks=[checkpoint])

    for it in range(5):
        seed = []
        seed.append(word_to_id['<START>'])
        seed.append(word_to_id['i'])
        seed.append(word_to_id['think'])
        seed.append(word_to_id['that'])

        for i in range(3, MAX_LEN):
            my_model.model.reset_states()
            seed_padded = sequence.pad_sequences([seed], maxlen=MAX_LEN, truncating='post', padding='post')
            y = my_model.model.predict(seed_padded)[0][i]
            #next_word_id = sample_multinomial(y, temperature=0.5)
            next_word_id = sample_argmax(y)
            seed.append(next_word_id)

        gen_sentence = ''
        for i in range(MAX_LEN):
            gen_sentence = gen_sentence + ' ' + id_to_word[seed[i]]

        print(gen_sentence)


if __name__ == '__main__':
    main()
