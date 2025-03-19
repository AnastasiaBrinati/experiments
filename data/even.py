import pandas as pd
import sys

def split_csv(input_file, output1, output2):
    # Load CSV
    df = pd.read_csv(input_file)

    # Split dataframe by alternating rows
    df_part1 = df.iloc[::2]  # Righe con indice pari
    df_part2 = df.iloc[1::2]  # Righe con indice dispari

    # Save to new files
    df_part1.to_csv(output1, index=False)
    df_part2.to_csv(output2, index=False)

    print(f"Split complete! Files saved as '{output1}' and '{output2}'")


if __name__ == "__main__":

    input_file = "data/trace/debs15_1.csv"
    split_csv(input_file, "data/trace/debs15_1_even.csv", "data/trace/debs15_1_odd.csv")
