import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal


class Critic:
    def __init__(self, state_shape=(1,), action_shape=(1,),
                 output_shape=(1,), learning_rate=0.002, seed=2,
                 discount_rate=0.95, tau=0.05, l2_weight=0.001):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.output_shape = output_shape
        self.tau = tau  # for update the target net
        self.lr = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.lr, clipnorm=100.)
        self.discount_rate = discount_rate
        self.action_grads = None
        self.initializer = he_normal(seed)
        self.regularizer = l2(l2_weight)
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        action = Input(self.action_shape)
        state = Input(self.state_shape)
        # state = Dense(32, activation='relu')(state)
        inp = concatenate([state, action], 1)
        inp = BatchNormalization()(inp)
        inp = Dense(200, activation='tanh', kernel_initializer=self.initializer)(inp)
        inp = Dense(100, activation='relu', kernel_initializer=self.initializer)(inp)
        # inp = Dense(50, activation='tanh', dtype='float64')(inp)
        out = Dense(self.output_shape[0], kernel_initializer=self.initializer)(inp)
        model = Model([state, action], out)
        model.compile(optimizer='adam',
                      loss='MSE')
        return model

    def train(self, x_train, y_train):
        with tf.GradientTape(persistent=True) as tape:
            # tape.watch(x_train)
            out = self.model(x_train)
            loss = tf.reduce_mean(1 / 2 * (y_train - out) ** 2)
        train_grads = tape.gradient(loss, self.model.trainable_variables)
        # train_grads = [tf.clip_by_norm(grad, 10.) for grad in train_grads]
        # self.action_grads = tf.reduce_mean(tape.gradient(out, x_train[1]))
        # self.action_grads = tf.clip_by_norm(self.action_grads, 100.)
        self.optimizer.apply_gradients(zip(train_grads, self.model.trainable_variables))

    def predict_target(self, inp):
        return self.target_model.predict_on_batch(inp)


class Actor:
    def __init__(self, input_shape=(1,), output_shape=(1,), action_range=[2],
                 seed=2, learning_rate=0.003, tau=0.05, l2_weight=0.001):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.action_range = action_range
        self.tau = tau
        self.lr = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.lr, clipnorm=100.)
        self.regularizer = l2(l2_weight)
        self.initializer = he_normal(seed)
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        inp0 = Input(self.input_shape)
        inp = BatchNormalization()(inp0)
        inp = Dense(200, activation='relu', kernel_initializer=self.initializer)(inp)
        inp = Dense(100, activation='relu', kernel_initializer=self.initializer)(inp)
        out = Dense(self.output_shape[0], activation='tanh',
                    kernel_initializer=self.initializer)(inp)
        out *= self.action_range
        model = Model(inp0, out)
        model.compile(optimizer='adam',
                      loss='MSE')
        return model

    def train(self, states, critic_model):
        with tf.GradientTape() as tape:
            actions = self.model(states)
            loss = -tf.reduce_mean(critic_model([states, actions]))
        grads = tape.gradient(loss, self.model.trainable_variables)
        # grads = [tf.clip_by_norm(grad, 10.) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict_target(self, inp):
        return self.target_model.predict_on_batch(inp)
    