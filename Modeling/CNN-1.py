import input_data
import eval_measures
import numpy as np
from keras.layers import Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Input, Dense
from keras.models import Model


MAX_SEQUENCE_LENGTH = 250  # max_len
MAX_NB_WORDS = 30000       # max_features
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.25
BATCH_SIZE = 64
EPOCHS = 10

print('Build model...')

main_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding_layer = Embedding(MAX_NB_WORDS,  # len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)(main_input)

conv1 = Conv1D(128, 5, activation='relu')(embedding_layer)
max1 = MaxPooling1D(5)(conv1)
conv2 = Conv1D(128, 5, activation='relu')(max1)
max2 = MaxPooling1D(5)(conv2)

flat = Flatten()(max2)
dense1 = Dense(128, activation='relu')(flat)
chapter_output = Dense(13, activation='softmax')(dense1)

cnn_1_model = Model(main_input, chapter_output)
cnn_1_model.summary()

cnn_1_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

cnn_1_model.fit(input_data.x_train,
                input_data.y_train_chapter,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True)


test_chapter = cnn_1_model.predict(input_data.x_test, batch_size=64)
test_chapter_bool = np.argmax(test_chapter, axis=1)

report_with_auc = input_data.class_report(y_true=np.argmax(input_data.y_test_chapter, axis=1),
                                           y_pred=test_chapter_bool,
                                           y_score=test_chapter)
print('Reports on test chapter')
print("single-label case:")
print('[loss, accuracy] = ')
print(cnn_1_model.evaluate(input_data.x_test, input_data.y_test_chapter, verbose=False))
print(report_with_auc)

print("multi-label case:")
print("accuracy, precision, recall, f-measure = ",
      eval_measures.eval_measures_multi_label(test_chapter,
                                              np.argmax(input_data.y_test_chapter, axis=1),
                                              input_data.patient_id_test, threshold=0.08, level='chapter'))


digit_output = Dense(56, activation='softmax')(dense1)
cnn_1_model = Model(main_input, digit_output)
cnn_1_model.summary()

cnn_1_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

cnn_1_model.fit(input_data.x_train,
                input_data.y_train_digit,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True)


test_digit = cnn_1_model.predict(input_data.x_test, batch_size=64)
test_digit_bool = np.argmax(test_digit, axis=1)

report_with_auc = input_data.class_report(y_true=np.argmax(input_data.y_test_digit, axis=1),
                                           y_pred=test_digit_bool,
                                           y_score=test_digit)
print('Reports on test digit')
print("single-label case:")
print('[loss, accuracy] = ')
print(cnn_1_model.evaluate(input_data.x_test, input_data.y_test_digit, verbose=False))
print(report_with_auc)

print("multi-label case:")
print("accuracy, precision, recall, f-measure = ",
      eval_measures.eval_measures_multi_label(test_digit,
                                              np.argmax(input_data.y_test_digit, axis=1),
                                              input_data.patient_id_test, threshold=0.009, level='rolledup'))


code_output = Dense(96, activation='softmax')(dense1)
cnn_1_model = Model(main_input, code_output)
cnn_1_model.summary()

cnn_1_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

cnn_1_model.fit(input_data.x_train,
                input_data.y_train_code,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True)


test_code = cnn_1_model.predict(input_data.x_test, batch_size=64)
test_code_bool = np.argmax(test_code, axis=1)

report_with_auc = input_data.class_report(y_true=np.argmax(input_data.y_test_code, axis=1),
                                           y_pred=test_code_bool,
                                           y_score=test_code)
print('Reports on test code')
print("single-label case:")
print('[loss, accuracy] = ')
print(cnn_1_model.evaluate(input_data.x_test, input_data.y_test_code, verbose=False))
print(report_with_auc)

print("multi-label case:")
print("accuracy, precision, recall, f-measure = ",
      eval_measures.eval_measures_multi_label(test_code,
                                              np.argmax(input_data.y_test_code, axis=1),
                                              input_data.patient_id_test, threshold=0.001, level='fullcode'))
