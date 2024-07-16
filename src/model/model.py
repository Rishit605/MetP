import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# class OtherLSTMModel(tf.keras.Model):
#     def __init__(self, n_lstm_units = 64, n_dense_units = 32, n_features=None, n_target_variables=None):
#         super(OtherLSTMModel, self).__init__()

#         self.lstm = layers.LSTM(n_lstm_units, activation='relu', return_sequences=False)
#         self.dense1 = layers.Dense(n_dense_units, activation='relu')
#         self.dense2 = layers.Dense(n_target_variables)

#         self.input_shape = None

#     def call(self, inputs):
#         x = self.lstm(inputs)
#         x = self.dense1(x)
#         return self.dense2(x)

#     def build(self,input_shape):
#         self.input_shape = input_shape
#         super(OtherLSTMModel, self).build(input_shape)


# class LSTMModel:
#     def __init__(self, input_shape, output_size, units=64, dropout_rate=0.2, l2_reg=0.001):
#         self.input_shape = input_shape
#         self.output_size = output_size
#         self.units = units
#         self.dropout_rate = dropout_rate
#         self.l2_reg = l2_reg
#         self.model = self.build_model()

#     def build_model(self):
#         model = Sequential()
#         model.add(LSTM(self.units, input_shape=self.input_shape, return_sequences=True))
#         model.add(Dropout(self.dropout_rate))
#         model.add(LSTM(self.units, return_sequences=True))
#         model.add(Dropout(self.dropout_rate))
#         model.add(Dense(len(self.output_size), activation='linear'))
#         return model

#     def compile_model(self, optimizer='adam', loss='mean_squared_error', metrics=['mean_squeared_error']):
#       self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

#     def fit(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None, callbacks=None):
#         if callbacks is None:
#             callbacks = self.get_default_callbacks()
#         history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
#                         validation_data=validation_data, callbacks=callbacks)
#         return history

#     def predict(self, X_test):
#         return self.model.predict(X_test)

#     def get_default_callbacks(self):
#         early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#         checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
#         tensorboard = TensorBoard(log_dir='./logs')
#         return [early_stopping, checkpoint, tensorboard]

#     def save_model(self, filepath):
#       self.model.save(filepath) 

class GRUModel:
    def __init__(self, input_shape, output_size, units=64, dropout_rate=0.33, l2_reg=0.001):
        self.input_shape = input_shape
        self.output_size = output_size
        self.units = units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        
        model.add(GRU(self.units, input_shape=self.input_shape))
        model.add(Dropout(self.dropout_rate))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(self.output_size), activation='linear'))
        return model

    def train_step(self, Train_Set, Valid_Set, epochs=50, batch_size=32, optim=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError(), metric=tf.keras.metrics.MeanSquaredError(), callbacks=None):
        if callbacks is None:
            callbacks = self.get_default_callbacks()
        
        train_loss, val_loss = [], []
        train_rmse, val_rmse = [], []

        for epoch in range(epochs):
            epoch_train_loss = []
            epoch_train_rmse = []

            for step, (X_batch_train, Y_batch_train) in enumerate(Train_Set):
                with tf.GradientTape() as Tape:
                    y_pred = self.model(X_batch_train, training=True)
                    loss_value = loss(Y_batch_train, y_pred)

                gradients = Tape.gradient(loss_value, self.model.trainable_variables)
                optim.apply_gradients(zip(gradients, self.model.trainable_variables))
                metric.update_state(Y_batch_train, y_pred)

                epoch_train_loss.append(loss_value.numpy())
                epoch_train_rmse.append(metric.result().numpy())

                if step % 100 == 0:
                    print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss_value.numpy()}")
                    print(f"Epoch {epoch + 1}, Step {step}, RMSE: {metric.result().numpy()}")
                    print(f"Seen so far: {(step + 1) * batch_size} samples")
        
        train_loss.append(sum(epoch_train_loss) / len(epoch_train_loss))
        train_rmse.append(sum(epoch_train_rmse) / len(epoch_train_rmse))

        epoch_val_loss = []
        epoch_val_rmse = []

        for X_batch_val, Y_batch_val in Valid_Set:
            y_pred = self.self.model(X_batch_val, training=False)
            loss_value = loss(Y_batch_val, y_pred)

            metric.update_state(Y_batch_val, y_pred)

            epoch_val_loss.append(loss_value.numpy())
            epoch_val_rmse.append(metric.result().numpy())

        val_loss.append(sum(epoch_val_loss) / len(epoch_val_loss))
        val_rmse.append(sum(epoch_val_rmse) / len(epoch_val_rmse))

        return train_loss, val_loss, train_rmse, val_rmse

    def get_default_callbacks(self):
        checkpoint_filepath = 'best_model.keras'
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )

        # Early Stopping Callback (Optional)
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,  # Stop if no improvement after 10 epochs
            restore_best_weights=True  # Restore weights from the best epoch
        )

        return model_checkpoint_callback, early_stopping_callback

# ---------------------------------------------------------------------- #


class WRFModel(tf.keras.Model):
    def __init__(self, n_lstm_units = 64, n_dense_units = 32, n_features=None, n_target_variables=None):
        super(WRFModel, self).__init__()

        self.lstm = layers.LSTM(n_lstm_units, activation='relu', return_sequences=False)
        self.dense1 = layers.Dense(n_dense_units, activation='relu')
        self.dense2 = layers.Dense(n_target_variables)

        self.input_shape = None

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense1(x)
        return self.dense2(x)

    def build(self,input_shape):
        self.input_shape = input_shape
        super(WRFModel, self).build(input_shape)