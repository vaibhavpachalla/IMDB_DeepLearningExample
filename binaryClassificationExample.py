#This program was written with guidance from the book Deep Learning with Python
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras import regularizers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#Arrays for plotting info
loss_values = []
loss_values_labels = []
val_loss_values = []
val_loss_values_labels = []



def vectorize_sequences(sequences, dimension=10000):
	results=np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

'''
#l1 regularization
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l1(0.001),  activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l1(0.001), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose = 0)

results = model.evaluate(x_test, y_test)
print(results)   



loss_values.append(history.history['loss'])
loss_values_labels.append('Training L1 loss')
val_loss_values.append(history.history['val_loss'])
val_loss_values_labels.append('Validation L1 loss')

#l2 regularization
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),  activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose = 0)

results = model.evaluate(x_test, y_test)
print(results)   



loss_values.append(history.history['loss'])
loss_values_labels.append('Training L2 loss')
val_loss_values.append(history.history['val_loss'])
val_loss_values_labels.append('Validation L2 loss')



#l1 and l2 simultaneous regularization
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l1_l2(l1 = 0.001, l2 = 0.001),  activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l1_l2(l1 = 0.001, l2 = 0.001), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose = 0)

results = model.evaluate(x_test, y_test)
print(results)   



loss_values.append(history.history['loss'])
loss_values_labels.append('Training simultaneous loss')
val_loss_values.append(history.history['val_loss'])
val_loss_values_labels.append('Validation simultaneous loss')
'''

#No Regularization
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose = 0)

results = model.evaluate(x_test, y_test)
print(results)   



loss_values.append(history.history['loss'])
loss_values_labels.append('Training No Reg loss')
val_loss_values.append(history.history['val_loss'])
val_loss_values_labels.append('Validation No Reg loss')



#Dropout Regularization
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose = 0)

results = model.evaluate(x_test, y_test)
print(results)   



loss_values.append(history.history['loss'])
loss_values_labels.append('Training Dropout Reg loss')
val_loss_values.append(history.history['val_loss'])
val_loss_values_labels.append('Validation Dropout Reg loss')





#Plot training and validation loss

#In future try not to have hard coded values
epochs = range(1, 20 + 1)

#Plot Training Loss
#plt.plot(epochs, loss_values[0], 'ro', label = loss_values_labels[0])
#plt.plot(epochs, loss_values[1], 'bo', label = loss_values_labels[1])
#plt.plot(epochs, loss_values[2], 'go', label = loss_values_labels[2])
#plt.plot(epochs, loss_values[3], 'co', label = loss_values_labels[3])
#plt.plot(epochs, loss_values[0], 'co', label = loss_values_labels[0])
#plt.plot(epochs, loss_values[1], 'mo', label = loss_values_labels[1])

#Plot Validation Loss
#plt.plot(epochs, val_loss_values[0], 'r', label = val_loss_values_labels[0])
#plt.plot(epochs, val_loss_values[1], 'b', label = val_loss_values_labels[1])
#plt.plot(epochs, val_loss_values[2], 'g', label = val_loss_values_labels[2])
#plt.plot(epochs, loss_values[3], 'c', label = val_loss_values_labels[3])
plt.plot(epochs, val_loss_values[0], 'c', label = val_loss_values_labels[0])
plt.plot(epochs, val_loss_values[1], 'm', label = val_loss_values_labels[1])


plt.title('Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



