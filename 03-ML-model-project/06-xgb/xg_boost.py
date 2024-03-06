import json

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def xg_boost(args):

    # Open and reads file "data"
    with open(args.data) as data_file:
        data = json.load(data_file)
    
    # Data type is 'dict', however since the file was loaded as a json object, it is first loaded as a string
    # thus we need to load again from such string in order to get the dict-type object.
    data = json.loads(data)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    
    # Initialize and train the model
    model = XGBClassifier()
    model.fit(x_train, y_train)

    # Get predictions
    y_pred = model.predict(x_test)
    
    # Get accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Save output into file
    with open(args.accuracy, 'w') as accuracy_file:
        accuracy_file.write(str(accuracy))

if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='My program description')
    # Input argument: data
    parser.add_argument('--data', type=str)
    # Output argument: accuracy
    parser.add_argument('--accuracy', type=str)

    args = parser.parse_args()

    # Creating the directory where the OUTPUT file will be created (the directory may or may not exist).
    Path(args.accuracy).parent.mkdir(parents=True, exist_ok=True)
    
    xg_boost(args)