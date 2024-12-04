from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import BatchNormalization

forecast_horizon = 1
sequence_length = 15

# Load X and y from .npy files
X = np.load("X_sequences.npy")
y = np.load("Y_targets.npy")
timestamps = np.load("timestamps.npy", allow_pickle=True)

# Determine the split index
split_index = int(0.95 * len(X))  # 95% for training, 5% for testing

# Temporal split
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
timestamps = timestamps[-len(y_test):]

# Define the model
model = Sequential([
    LSTM(15, activation='relu', input_shape=(sequence_length, X.shape[2])),
    #BatchNormalization(),
    Dense(forecast_horizon)
])

"""

attivazione: relu || linear || tanh

epoche: 20 || 10 || 25

layers: 1 layer: 15 neuroni

batchnormlization: on || off

"""


# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model and store training history
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Save loss progress graph
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Progress During Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_progress.png")  # Save the graph as a PNG image
plt.close()  # Close the figure to free memory

# Predict and compare with actual values
y_pred = model.predict(X_test)

# Save bck everythin you will need for the graph
#np.save("predictions.npy", y_pred)
np.save("timestamps.npy", timestamps)
#np.save("actuals.npy", y_test)

# Combine the arrays column-wise (actuals + predictions)
combined = np.column_stack((y_test, y_pred))
# Create a DataFrame from the combined array
df = pd.DataFrame(combined, columns=["actuals", "predictions"])
df.to_csv("data/results.csv", index=False)