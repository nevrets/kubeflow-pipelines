import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from io import StringIO


def load_data(data):
    d = StringIO(data)
    iris = pd.read_csv(d, sep=',')
    print(iris.shape)
    
    return iris

def get_train_test_data(iris):
    le = LabelEncoder()
    iris['Species'] = le.fit_transform(iris['Species'])
    
    train, test = train_test_split(iris, test_size=0.2, random_state=42)
    print('Shape of train data: ', train.shape)
    print('Shape of test data: ', test.shape)
    
    X_train = train.drop(columns=['Species'], axis=1)
    y_train = train['Species']
    X_test = test.drop(columns=['Species'], axis=1)
    y_test = test['Species']
    
    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('--data', type=str, help="Input data csv")

    config = p.parse_args()
    iris = config.data         # 01-data-loading 부분에서 load된 데이터를 받기 위해 argument 설정.
    iris = load_data(iris)     # 저장된 데이터를 해당 pipeline 단계에서 사용하려면 StringIO로 변환해주어야 함.
    
    X_train, X_test, y_train, y_test = get_train_test_data(iris)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)