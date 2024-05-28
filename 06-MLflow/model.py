import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.sklearn import save_model
from mlflow.tracking.client import MlflowClient

import os

# kubeflow - minio-service - 9000:30234/TCP  
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://master-node-ip:port"    # "http://minio-service.kubeflow.svc:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

mlflow.set_tracking_uri("http://master-node-ip:port")    # "http://mlflow-service.mlflow-system.svc:5000"
# mlflow-system  mlflow-service  5000:32677/TCP

experiment = mlflow.get_experiment(experiment_id="0")

def main():
    iris = load_iris()

    data = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    target = pd.DataFrame(iris["target"], columns=["target"])

    clf = SVC(kernel="rbf")
    clf.fit(data, target)

    input_example = data.sample(1)
    signature = infer_signature(data, clf.predict(data))
    conda_env = _mlflow_conda_env(additional_pip_deps=["dill", "pandas", "scikit-learn"])

    save_model(
        sk_model=clf,
        path="06-MLflow/svc",
        serialization_format="cloudpickle",
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
    )
    
    client = MlflowClient("http://172.7.0.45:32677")
    run = client.create_run(experiment_id='38')
    client.log_artifact(run.info.run_id, '06-MLflow/svc')


if __name__ == '__main__':
    main()