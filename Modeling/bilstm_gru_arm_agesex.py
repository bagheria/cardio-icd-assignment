import input_data
import mlabels_eval_measures
import numpy as np
from keras.layers import Embedding, concatenate
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import LSTM, SpatialDropout1D, GRU
from keras.layers import Bidirectional, Concatenate
from keras.layers import GlobalAveragePooling1D, Dropout
from sklearn.metrics import precision_recall_fscore_support


def dl_multi_input(n_input_variables=150, n_add_variables=2, n_output=10):
    reports_input = Input(shape=(n_input_variables,), dtype='int32', name='letters_input')
    x = Embedding(output_dim=512, input_dim=10000, input_length=n_input_variables)(reports_input)
    lstm_out = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2))(x)

    medical_vars_input = Input(shape=(n_add_variables,), name='vars_input')
    x = concatenate([lstm_out, medical_vars_input])

    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    final_output = Dense(n_output, activation='softmax', name='final_output')(x)
    model = Model(inputs=[reports_input, medical_vars_input], outputs=final_output)

    return model


MAX_SEQUENCE_LENGTH = 156  # max_len
MAX_NB_WORDS = 20000       # max_features
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.20
BATCH_SIZE = 64
EPOCHS = 1

print('Build model...')

model = dl_multi_input(n_input_variables=MAX_SEQUENCE_LENGTH,
                       n_add_variables=2,
                       n_output=10)
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit({'letters_input': input_data.x_train, 'vars_input': input_data.vars_dataframe},
          {'final_output': input_data.y_train_code},
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)

labels_pred = model.predict([input_data.x_train, input_data.vars_dataframe])

test_code_bool = np.argmax(labels_pred, axis=1)

items = "E11", "E78", "I10", "I21", "I25", "I42", "I48", "I50", "N18", "Z95"

print("precision, recall, f-measure = ",
      precision_recall_fscore_support(np.argmax(input_data.y_train_code, axis=1), test_code_bool, average='micro'))

print('Reports on test code')
print("multi-label case:")
threshold = 0.07
for t in range(10):
    print("threshold", threshold)
    print("\n sensitivity, specificity, ppv, npv, f1_score = ",
          mlabels_eval_measures.eval_measures_multi_label(labels_pred,
                                                  items,
                                                  input_data.patient_id_train, threshold=threshold, level='icdcode'))
    threshold += 0.01

print("Done!")
