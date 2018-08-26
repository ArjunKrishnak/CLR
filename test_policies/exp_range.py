import matplotlib.pyplot as plt

### CLR CALLBACK ###
from clr_callback import *
from keras.optimizers import *

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input

### DUMMY TRAINING DATA ###
inp = Input(shape=(15,))                
x = Dense(10, activation='relu')(inp)
x = Dense(1, activation='sigmoid')(x)

model = Model(inp, x)

X = np.random.rand(2000000,15) # Creates a matrix of 2000000*15 with random values
Y = np.random.randint(0,2,size=2000000) # Creates an array of 2000000 elements with random integers 0 or 1

### SGD ###
model.compile(optimizer=SGD(0.1), loss='binary_crossentropy', metrics=['accuracy'])
#model.fit(X, Y, batch_size=2000, epochs=10)

clr_exp_range = CyclicLR(mode='exp_range', gamma=0.9997)
model.fit(X, Y, batch_size=2000, epochs=12, callbacks=[clr_exp_range], verbose=1)

plt.xlabel('Training Iterations')
plt.ylabel('Learning Rate')
plt.title("CLR - 'exp_range' Policy")
plt.plot(clr_exp_range.history['iterations'], clr_exp_range.history['lr'])
plt.savefig('test_images/exp_range.png')
plt.show()