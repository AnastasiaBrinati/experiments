import pandas as pd
import matplotlib.pyplot as plt


def plot_cumulative_mean(file_paths, labels, output_path):
    """
    Plots cumulative means from multiple CSV files.

    Args:
        file_paths (list): List of file paths to the CSV files.
        labels (list): List of labels for each line in the plot.
        output_path (str): File path to save the plot image.
    """
    plt.figure(figsize=(10, 6))

    for file_path, label in zip(file_paths, labels):
        # Leggere il file CSV
        data = pd.read_csv(file_path, header=None)

        # Assumendo che la colonna di interesse sia la prima (colonna 0)
        column = data.iloc[:, 0]

        # Calcolare la media cumulativa
        cumulative_mean = column.expanding().mean()

        # Plottare la linea
        plt.plot(cumulative_mean, label=label)

    # Personalizzare il grafico
    plt.title('Cumulative Mean Plot')
    plt.xlabel('Index')
    plt.ylabel('Cumulative Mean')
    plt.legend()
    plt.grid(True)

    # Salvare il grafico come immagine
    plt.savefig(output_path)
    print(f"Grafico salvato in: {output_path}")

if __name__ == "__main__":
    # Percorsi dei file CSV
    file_paths = [
        'data/traces/endpoint0/functions/inter_arrivals0.csv',
        'data/traces/endpoint0/functions/inter_arrivals1.csv',
        'data/traces/endpoint1/functions/inter_arrivals0.csv',
        'data/traces/endpoint1/functions/inter_arrivals1.csv'
    ]

    # Etichette per le linee del grafico
    labels = ['E0-F0', 'E0-F1', 'E1-F0', 'E1-F1']

    # Percorso per salvare il grafico
    output_path = 'cumulative_mean_plot.png'

    # Generare il grafico
    plot_cumulative_mean(file_paths, labels, output_path)
