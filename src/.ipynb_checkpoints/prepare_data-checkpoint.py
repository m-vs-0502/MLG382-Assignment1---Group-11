import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare():
    # 1. Step out of notebooks to find the original file
    # Ensure this filename matches your folder exactly!
    input_path = '../data/Diabetes_and_LifeStyle_Dataset_.csv'
    
    # 2. Step out of notebooks to find the save location
    output_folder = '../data/'
    
    # Safety check: if the folder still isn't found, this creates it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(input_path)
    
    # Split the data
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # 3. Save using the relative path
    train.to_csv(os.path.join(output_folder, 'train.csv'), index=False)
    test.to_csv(os.path.join(output_folder, 'test.csv'), index=False)
    
    print("Files saved successfully in the project's data folder.")

if __name__ == "__main__":
    prepare()