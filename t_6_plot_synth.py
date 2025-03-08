import matplotlib.pyplot as plt
import pandas as pd

def plot(i):
    df = pd.read_csv("data/traces/endpoint"+str(i)+"/synthetic/arrival_rates_120s.csv")
    #synth_df = pd.read_csv("data/traces/endpoint"+str(i)+"/synthetic/inter_arrivals"+str(i)+".csv")

    # Plot
    plt.figure(figsize=(21, 7))
    plt.plot(df.iloc[:, 0], linestyle='-', color='blue', label='Actual')
    #plt.plot(synth_df.iloc[:, 0], linestyle='--', color='red', label='Synthetic')

    # Labels and Title
    plt.xlabel("update events")
    plt.ylabel("rates")
    plt.title("Synthetic rates")
    plt.legend()
    plt.grid(True)

    # Show Plot
    plt.savefig("data/traces/endpoint"+str(i)+"/synthetic/e"+str(i)+".png")

if __name__ == "__main__":
    for i in range(0,2):
        print(f"working on endpoint: {i}")
        plot(i)