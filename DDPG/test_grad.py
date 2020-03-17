import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, concatenate

with tf.device('/CPU:0'):
    tf.keras.backend.set_floatx('float32')
    action = Input((1,))
    state = Input((2,))
    state1 = Dense(16, activation='relu')(state)
    action1 = Dense(16, activation='relu')(action)
    # state = Dense(32, activation='relu')(state)
    inp = concatenate([state1, action1], 1)
    # print(type(inp))
    inp = Dense(48, activation='relu')(inp)
    inp = Dense(32, activation='linear')(inp)
    out = Dense(1)(inp)
    model = Model([state, action], out)
    model.compile(optimizer='adam',
                  loss='MSE')
    print(model.summary())
    # model.fit([np.array([[3], [4]]), np.array([[1, 2], [3, 4]])], np.array([[-1], [1]]))
    inp = [tf.convert_to_tensor(np.array([[1., 2.], [3., 4.]]), dtype='float32'),
           tf.convert_to_tensor(np.array([[3.], [4.]]), dtype='float32')]
    # inp = [np.array([[1., 2.], [3., 4.]]), np.array([[3.], [4.]])]
    y_train = np.array([[3.], [4.]])
    with tf.GradientTape() as tape:
        tape.watch(inp)
        out = model(inp)
        loss = tf.reduce_mean(1/2 * (y_train-out)**2)
    grad1 = tape.gradient(out, inp[1])
    # optimizer = tf.keras.optimizers.Adam()
    print(grad1)
    # optimizer.apply_gradients(zip(grad1, model.trainable_variables))
