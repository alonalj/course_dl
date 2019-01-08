import numpy as np
import pdb
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.layers import Bidirectional


# load the dataset but only keep the top words, zero the rest. Introduce special tokens.
top_words = 5000
(X_train, y_train), _= imdb.load_data(num_words=top_words)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                    test_size=0.5, 
                                                    stratify=y_train)

word_to_id = imdb.get_word_index()
word_to_id = {k: (v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<OOV>"] = 2
id_to_word = {v: k for k, v in word_to_id.items()}

max_review_length = 100
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length, truncating='post')
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, truncating='post')

embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Bidirectional(LSTM(60, dropout=0.6, recurrent_dropout=0.1, return_sequences=True)))
model.add(LSTM(20, dropout=0.3))
model.add(Dense(1, activation='sigmoid')) # We are dealing with binary classification so softmax isn't needed
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, 
          validation_data=(X_test, y_test), 
          epochs=7, batch_size=128)

