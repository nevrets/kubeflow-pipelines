import os
import pandas as pd
import argparse


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default='/data/nevret/kubeflow_example/Iris/01-data-loading')
    
    config = p.parse_args()
    
    return config

def get_dataframe_from_local(config):
    data = pd.read_csv(os.path.join(config.data_path, 'Iris.csv'))
    data.to_csv('./iris.csv', index=False)
    
    return data



if __name__ == '__main__':
    config = get_config()
    
    print("Load data")
    data = get_dataframe_from_local(config)
    print(data.shape)
    
    
    
    