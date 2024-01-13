import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    ionosphere = pd.read_csv("data/ionosphere.csv")  # Assuming the CSV file is in the data directory

    # Convert to table
    ionosphere['X35'] = ionosphere['class'].map({'g': 98, 'b': 103})
    Xdataset = ionosphere.drop(columns=['class'])

    # Split the data into training and validation sets
    train_data, validation_data = train_test_split(Xdataset, test_size=0.2, random_state=42)

    return train_data, validation_data
