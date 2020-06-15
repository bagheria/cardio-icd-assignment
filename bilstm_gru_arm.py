import input_data
import mlabels_eval_measures
import numpy as np
from keras.layers import Embedding
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import SpatialDropout1D, GRU
from keras.layers import Bidirectional, Concatenate
from keras.layers import GlobalAveragePooling1D, Dropout
from sklearn.metrics import precision_recall_fscore_support


MAX_SEQUENCE_LENGTH = 936  # avg_len
MAX_NB_WORDS = 10000       # max_features
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.20
BATCH_SIZE = 64
EPOCHS = 1

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
output = Dense(10, activation='softmax')(dense1)

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

items = "E11", "E78", "I10", "I21", "I25", "I42", "I48", "I50", "N18", "Z95"

print("precision, recall, f-measure = ",
      precision_recall_fscore_support(np.argmax(input_data.y_train_code, axis=1), test_code_bool, average='micro'))

print('Reports on test code')
print("multi-label case:")
threshold = 0.08
for t in range(10):
    print("threshold", threshold)
    print("\n sensitivity, specificity, ppv, npv, f1_score = ",
          mlabels_eval_measures.eval_measures_multi_label(test_code,
                                                  items,
                                                  input_data.patient_id_train, threshold=threshold, level='icdcode'))
    threshold += 0.01

print("Done!")
