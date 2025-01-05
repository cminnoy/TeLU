import tensorflow as tf

class TeLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TeLU, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.tanh(tf.exp(inputs))

# Example usage in a model
if __name__ == "__main__":
    # Define a simple model to test TeLU
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),  # Input layer
        tf.keras.layers.Dense(32),           # Dense layer
        TeLU(),                              # Custom activation
        tf.keras.layers.Dense(1)             # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Dummy data for testing
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)

    # Train the model
    model.fit(X, y, epochs=5, batch_size=10)

    # Print a summary
    model.summary()
