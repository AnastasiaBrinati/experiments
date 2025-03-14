import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

timestamps = np.load("timestamps.npy", allow_pickle=True)

df = pd.read_csv("data_globus/results/results.csv")
y_act = df["actuals"]
y_pred = df["predictions"]

# Save actual vs predicted graph
plt.figure(figsize=(25, 6))
plt.plot(y_act, label="Actual", marker="o")
plt.plot(y_pred, label="Predicted", marker="x")
plt.title(f"Actual vs Predicted (Forecast Step {1})")
plt.xlabel("Timestamps")
plt.ylabel("N° invocations")
plt.xticks(ticks=range(len(timestamps)), labels=timestamps, rotation=45)
plt.legend()
plt.grid()
plt.savefig(f"actual_vs_predicted_step_{1}.png")  # Save for each forecast step
plt.close()  # Close the figure to free memory
