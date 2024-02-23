import os
import argparse
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from io import StringIO

from kfp import dsl


def get_config():
    p = argparse.ArgumentParser()
    
    p.add_argument('--data', type=str, default="/data/nevret/kubeflow_example/02-Iris/logistic_regression/Iris.csv")
    p.add_argument('--dir_path', type=str, default="/data/nevret/kubeflow_example/02-Iris/logistic_regression")
    
    config = p.parse_args()
    
    return config

def load_data(config):
    # d = StringIO(config.data)
    iris = pd.read_csv(os.path.join(config.dir_path, 'Iris.csv'), sep=',')
    print(iris.shape)
    
    return iris

@dsl.component
def logistic_regression():
    iris = load_data(config)     # 저장된 데이터를 해당 pipeline 단계에서 사용하려면 StringIO로 변환해주어야 함.
    
    le = LabelEncoder()
    iris['Species'] = le.fit_transform(iris['Species'])
    
    train, test = train_test_split(iris, test_size=0.2, random_state=42)
    print('Shape of train data: ', train.shape)
    print('Shape of test data: ', test.shape)
    
    # Split dataset
    X_train = train.drop(columns=['Species'], axis=1)
    y_train = train['Species']
    X_test = test.drop(columns=['Species'], axis=1)
    y_test = test['Species']
    
    # Train
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    
    # Accuracy
    print(f"Accuracy: ", score)
    
    # Save output into file
    with open(os.path.join(config.dir_path, 'score.txt'), 'w') as score_file:
        score_file.write(str(score))
    
    return score



if __name__ == '__main__':
    config = get_config()
    logistic_regression(config)
    
    