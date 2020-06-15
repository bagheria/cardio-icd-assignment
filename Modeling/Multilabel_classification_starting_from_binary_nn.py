import numpy as np
from keras.layers import Embedding
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import SpatialDropout1D, GRU
from keras.layers import Bidirectional, Concatenate
from keras.layers import GlobalAveragePooling1D, Dropout
import tensorflow as tf
import input_data


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


MAX_SEQUENCE_LENGTH = 400  # avg_len
MAX_NB_WORDS = 10000       # max_features
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.20
BATCH_SIZE = 64
EPOCHS = 10

print('Build model...')

main_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding_layer = Embedding(MAX_NB_WORDS,  # len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)(main_input)

drop0 = SpatialDropout1D(0.1)(embedding_layer)
rnn1 = Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.2))(drop0)


avgEmbedding = GlobalAveragePooling1D()(embedding_layer)
conc = Concatenate(axis=1)([rnn1, avgEmbedding])
conc_drop = Dropout(0.05)(conc)
dense1 = Dense(128, activation='relu')(conc_drop)
output = Dense(2, activation='softmax')(dense1)

model = Model(main_input, output)
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(input_data.x_train,
          input_data.y_train_code,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          shuffle=True)

test_code = model.predict(input_data.x_train, batch_size=64)
test_code_bool = np.argmax(test_code, axis=1)

print("done")
