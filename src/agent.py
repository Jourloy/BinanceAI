from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0001, gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.999, target_update_freq=10, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.timestep = 0
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        input_layer = Input(shape=self.state_size)
        x = Dense(32, activation="relu")(input_layer)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(32, activation="relu")(x)
        output_layer = Dense(self.action_size, activation="linear")(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=self.learning_rate))
        return model
