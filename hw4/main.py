import numpy as np
import pdb
import string
import argparse
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.layers import CuDNNLSTM, Bidirectional, Dropout, TimeDistributed, BatchNormalization, Input
from keras.utils import to_categorical
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import concatenate, add
from nltk.translate.bleu_score import corpus_bleu

IMDB_VOCABULARY_SIZE = 20000
CHAR_VOCABULARY_SIZE = 101
MAX_SAMPLES = 40000
MAX_LEN = 100
CHAR_MAX_LEN = 300

# Softmax sampling methods
# ------------------------
def sample_multinomial(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def sample_argmax(preds):
    return np.argmax(preds)

# Generate WL sentence
# --------------------
def generate_sentence_wl(model, word_to_id, id_to_word, technique, sentiment=None, diversity=0.7):
    seed = []
    seed.append(word_to_id['<START>'])
    seed.append(word_to_id['the'])
    seed.append(word_to_id['movie'])
    seed.append(word_to_id['was'])

    sentim_neg = np.zeros((1, MAX_LEN, 1))  # negative
    sentim_pos = np.ones((1, MAX_LEN, 1))  # positive

    for i in range(len(seed) - 1, MAX_LEN-1):
        if sentiment == 'pos':
            sentim = sentim_pos
        elif sentiment == 'neg':
            sentim = sentim_neg
        elif sentiment == 'neg2pos':
            if i < 50:
                sentim = sentim_neg
            else:
                sentim = sentim_pos
        elif sentiment is not None:
            raise ValueError('Sentiment can only get "pos", "neg", or "neg2pos" values.')

        model.reset_states()
        seed_padded = sequence.pad_sequences([seed], maxlen=MAX_LEN, truncating='post', padding='post')

        if sentiment is not None:
            y = model.predict([seed_padded, sentim])[0][i]
        else:
            y = model.predict(seed_padded)[0][i]

        if technique == 'multinomial':
            next_word_id = sample_multinomial(y, temperature=diversity)
        else:
            next_word_id = sample_argmax(y)
        seed.append(next_word_id)

    gen_sentence = ''
    for i in range(MAX_LEN-1):
        if i == 50 and sentiment == 'neg2pos':
            gen_sentence = gen_sentence + ' [NEG2POS] '
        gen_sentence = gen_sentence + ' ' + id_to_word[seed[i]]

    return gen_sentence


def evaluate_model(model, word_to_id, id_to_word, technique, X2_test, X_test):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in enumerate(X_test):
        print('{}/{}'.format(key, 256))
        # generate description

        if X2_test[key][0][0] == 1:
            sentiment = 'pos'
        else:
            sentiment = 'neg'

        yhat = generate_sentence_wl(model, word_to_id, id_to_word, technique, sentiment)
        # store actual and predicted

        gen_sentence = ''
        for i in range(MAX_LEN):
            gen_sentence = gen_sentence + ' ' + id_to_word[desc_list[i]]

        actual.append(gen_sentence.split())
        predicted.append(yhat.split())

        if key == 256:
            break

    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))



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

        self.filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-' + str(self._type).lower() +\
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

        elif self._type == 'WL-S':
            input1 = Input(shape=(MAX_LEN,))
            se1 = Embedding(IMDB_VOCABULARY_SIZE, self._embed_size, input_length=MAX_LEN, mask_zero=True)(input1)
            se2 = Dropout(self._do_ratio)(se1)

            input2 = Input(shape=(MAX_LEN, 1))
            concat1 = concatenate([se2, input2])

            # Stack 1
            lstm1 = LSTM(self._lstm_size, return_sequences=True)(concat1)
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
            output = TimeDistributed(Dense(IMDB_VOCABULARY_SIZE, activation='softmax'))(do5)

            model = Model(inputs=[input1, input2], outputs=output)
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


parser = argparse.ArgumentParser(description='HW4: 236606, Deep Learning Course, Winter 2018\n'
                                             'Alona Levy and Gil Shomron',
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-t', '--type', default='WL-S', type=str,
                    help='WL-S: word-level with sentiment\n'
                         'WL: word-level method without sentiment\n'
                         'CL: character-level method without sentiment')

parser.add_argument('--technique', default='multinomial', type=str,
                    help='argmax; multinomial')

parser.add_argument('--pretrained', default=None, type=str,
                    help='The path to the pretrained weights')

parser.add_argument('-n', default=10, type=int,
                    help='Number of sentences to produce.')

parser.add_argument('--bleu', action='store_true',
                    help='Run BLEU check, only relevant for WL-S model')

parser.add_argument('--sentiment', default='neg2pos', type=str,
                    help='Only relevant for WL-S model: neg2pos; neg; pos')

parser.add_argument('--diversity', default=0.7, type=float,
                    help='Only relevant for WL-S model: neg2pos; neg; pos')

def main():
    args = parser.parse_args()

    pretrained = args.pretrained
    type = args.type
    n_sentences = args.n

    if type != 'WL' and type != 'CL' and type != 'WL-S':
        raise ValueError('Type can only get WL (word-level), CL (character-level), or WL-S (word-level sentiment).')
    if args.technique != 'multinomial' and args.technique != 'argmax':
        raise ValueError('Technique can only get argmax or multinomial.')
    if args.sentiment != 'neg2pos' and args.sentiment != 'neg' and args.sentiment != 'pos':
        raise ValueError('Sentiment can only get neg2pos, neg, or pos.')

    print("Parameters:")
    print("> Model type: {}".format(type))
    if type == 'WL-S':
        print(">> Sentiment: {}".format(args.sentiment))
    print("> Softmax choice: {}".format(args.technique))
    if args.technique == 'multinomial':
        print(">> Diversity: {}".format(args.diversity))
    print("> Run BLEU: {}".format(args.bleu))
    print("> Pretrained: {}".format(pretrained))
    print("> Generate {} sentences".format(n_sentences))
    print("")

    # Load and preprocess the dataset
    # -------------------------------
    (sentences, sentiment), _ = imdb.load_data(num_words=IMDB_VOCABULARY_SIZE)

    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<OOV>"] = 2
    word_to_id["<EOS>"] = 3
    id_to_word = {v: k for k, v in word_to_id.items()}

    print("Parsing IMDB")
    if type == 'WL':
        X_train = sequence.pad_sequences(sentences, maxlen=MAX_LEN, truncating='post')
        # Desired output is a skewed input
        y_train = np.roll(X_train, -1, axis=-1)
        y_train = y_train[:, :, np.newaxis]

        X_train, X_test, y_train, y_test = train_test_split(np.array(X_train), np.array(y_train), test_size=0.1)

    elif type == 'WL-S':
        X_train = sequence.pad_sequences(sentences, maxlen=MAX_LEN, truncating='post')

        # The sentiment of the sentence is also part of the input
        X2_train = []
        for val in sentiment:
            X2_train.append(np.ones(MAX_LEN)*val)
        X2_train = np.array(X2_train)
        X2_train = X2_train[:, :, np.newaxis]

        y_train = np.roll(X_train, -1, axis=-1)
        y_train = y_train[:, :, np.newaxis]

        X_train, X_test, X2_train, X2_test, y_train, y_test =\
            train_test_split(np.array(X_train), np.array(X2_train), np.array(y_train), test_size=0.1)

    elif type == 'CL':
        characters = ['<PAD>', '<START>', '<EOS>'] + list(string.printable)
        characters.remove('\x0b')
        characters.remove('\x0c')
        assert (len(characters) == CHAR_VOCABULARY_SIZE)
        char2ind = {c: i for i, c in enumerate(characters)}

        # Words to characters conversion
        X_train_words = sequence.pad_sequences(sentences, maxlen=MAX_LEN, truncating='post')
        X_train = []
        for i, s in enumerate(X_train_words):
            X_train.append([])
            for w in s:
                word = id_to_word[w]
                for c in word:
                    try:
                        X_train[i].append(char2ind[c])
                    # Skipping unknown signs
                    except:
                        print("Skipping {} character".format(c))

                X_train[i].append(char2ind[' '])

        X_train = sequence.pad_sequences(X_train, maxlen=CHAR_MAX_LEN, truncating='post', padding='post')
        y_train = np.roll(X_train, -1, axis=-1)

        X_train = np.array([to_categorical(x, num_classes=CHAR_VOCABULARY_SIZE) for x in X_train])
        y_train = np.array([to_categorical(y, num_classes=CHAR_VOCABULARY_SIZE) for y in y_train])

        X_train, X_test, y_train, y_test = train_test_split(np.array(X_train), np.array(y_train), test_size=0.1)

    # Model and training
    # ------------------
    print("Creating model")
    my_model = TextGenModel(type=type, pretrained=pretrained, cl_in_shape=X_train.shape[1:])

    if pretrained is None:
        print("Start training")
        checkpoint = ModelCheckpoint(my_model.filepath, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')

        if type == 'WL' or type == 'CL':
            my_model.model.fit(X_train, y_train,
                               validation_data=(X_test, y_test), epochs=30,
                               batch_size=128, callbacks=[checkpoint])
        elif type == 'WL-S':
            my_model.model.fit([X_train, X2_train], y_train,
                               validation_data=([X_test, X2_test], y_test), epochs=50,
                               batch_size=128, callbacks=[checkpoint])

    # BLEU validation
    # ---------------
    if args.bleu:
        print("Running BLEU, stay tuned, this may take a while")
        evaluate_model(my_model.model, word_to_id, id_to_word, args.technique, X2_test, X_test)

    # Create sentences
    # ----------------
    print("Creating sentences")
    if type == 'WL':
        for it in range(n_sentences):
            seq = generate_sentence_wl(my_model.model, word_to_id, id_to_word, args.technique,
                                       diversity=args.diversity)
            print(seq)

    if type == 'WL-S':
        for it in range(n_sentences):
            seq = generate_sentence_wl(my_model.model, word_to_id, id_to_word, args.technique,
                                       args.sentiment, diversity=args.diversity)
            print(seq)

    elif type == 'CL':
        for it in range(n_sentences):
            diversity = args.diversity
            seed = "i think that "
            result = np.zeros((1,) + X_train.shape[1:])
            result[0, 0, char2ind['<START>']] = 1
            next_res_ind = 1
            for s in seed:
                result[0, next_res_ind, char2ind[s]] = 1
                next_res_ind = next_res_ind + 1

            print("[" + seed + "]", end='')

            next_char = seed[-1]
            while next_char != '<EOS>' and next_res_ind < CHAR_MAX_LEN:
                my_model.model.reset_states()
                y = my_model.model.predict_on_batch(result)[0][next_res_ind-1]
                if args.technique == 'multinomial':
                    next_char_ind = sample_multinomial(y, temperature=diversity)
                else:
                    next_char_ind = sample_argmax(y)
                next_char = characters[next_char_ind]
                result[0, next_res_ind, next_char_ind] = 1
                next_res_ind = next_res_ind + 1
                print(next_char, end='')
            print()


if __name__ == '__main__':
    main()
