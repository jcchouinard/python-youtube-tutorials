import glob
import pandas as pd

# Define the folder path and pattern for pickle files
folder_path = 'path/to/data'
pattern = f"{folder_path}/*.pkl"  # Pattern to match all .pkl files in the folder

# Use glob to find all pickle files
pickle_files = glob.glob(pattern)

# Read each pickle file into a DataFrame and store them in a list
dataframes = []
for file in pickle_files:
    try:
        x = pd.read_pickle(file)
        dataframes.append(x)
        print(f"Successfully loaded {file}")
    except Exception as e:
        print(f"Failed to load {file}: {e}")


master = pd.concat(dataframes)
master.head(30)