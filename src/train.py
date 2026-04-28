import tensorflow as tf
from tensorflow import keras

# Sample model and data
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Dummy data
import numpy as np
x_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)

# Training loop
for epoch in range(100):
    print(f'Iteration {epoch}')
    model.fit(x_train, y_train, epochs=1)
    # Optionally, you could add validation and logging here.
    
    # Optimizer step already handled in `model.fit`
    # Additional optimization like learning rate scheduling can also be added.