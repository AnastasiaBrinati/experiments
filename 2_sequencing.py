import numpy as np
import pandas as pd

# Simulazione di un dataset con giorni e feature

df = pd.read_csv("data/globus/300/globus.csv")
timestamps = df["timestamp"].to_numpy()
df.drop("timestamp", axis=1, inplace=True)

# Parametri
sequence_length = 15  # Numero di giorni nella sequenza
forecast_horizon = 1  # Numero di giorni da predire

# Funzione per creare dataset sequenziale
def create_sequences(data, target_column, sequence_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        seq_x = data.iloc[i:i + sequence_length].drop(target_column, axis=1).values
        seq_y = data.iloc[i + sequence_length:i + sequence_length + forecast_horizon][target_column].values
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Creazione del dataset sequenziale
X, y = create_sequences(df, target_column="n_invocations", sequence_length=sequence_length, forecast_horizon=forecast_horizon)

# Visualizzazione dei risultati
#print("Shape of X (Input Sequences):", X.shape)
#print("Shape of y (Target Values):", y.shape)

# Save X and y as .npy files
np.save("X_sequences.npy", X)
np.save("Y_targets.npy", y)
np.save("timestamps.npy", timestamps)