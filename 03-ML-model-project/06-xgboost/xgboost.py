import json

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def xgboost(config):
    # Open and reads file "data"
    with open(config.data) as data_file:
        data = json.load(data_file)
    
    # Data type is 'dict', however since the file was loaded as a json object, it is first loaded as a string
    # thus we need to load again from such string in order to get the dict-type object.
    data = json.loads(data)

    X_train = data['x_train']
    y_train = data['y_train']
    X_test = data['x_test']
    y_test = data['y_test']
    
    # Initialize and train the model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Save output into file
    with open(config.accuracy, 'w') as accuracy_file:
        accuracy_file.write(str(accuracy))
        
        
        
if __name__ == '__main__':
    
    # This component does not receive any input, it only outputs one artifact which is `data`.
    # Output argument: data
    p = argparse.ArgumentParser(description='Program description')
    p.add_argument('--data', type=str)
    p.add_argument('--accuracy', type=str)
    
    config = p.parse_args()
    
    # Creating the directory where the OUTPUT file will be created, (the directory may or may not exist).
    # This will be used for other component's input (e.g. decision tree, logistic regression)
    Path(config.data).parent.mkdir(parents=True, exist_ok=True)
    
    xgboost(config)