import pandas as pd
import sys

def split_csv(input_file, output1, output2):
    # Load CSV
    df = pd.read_csv(input_file)

    # Calculate split point
    mid_index = len(df) // 10

    # Split the dataframe
    df_part1 = df.iloc[:mid_index]
    df_part2 = df.iloc[mid_index:mid_index*2]

    # Save to new files
    df_part1.to_csv(output1, index=False)
    df_part2.to_csv(output2, index=False)

    print(f"Split complete! Files saved as '{output1}' and '{output2}'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_csv.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    split_csv(input_file, "data/trace/debs15_1.csv", "data/trace/debs15_2.csv")
