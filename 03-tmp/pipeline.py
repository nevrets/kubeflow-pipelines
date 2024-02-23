from kfp import dsl
import kfp.components as comp


@dsl.pipeline(
    name='nevret-iris',
    description='nevret kubeflow test - iris'
)

def iris_pipeline():
    data_preprocess = dsl.ContainerSpec(
        # name="load iris data pipeline",
        image="nevret/nevret-iris-preprocessing:0.5",
        command=['python'],
        args=[
            '--data_path', './iris.csv'
        ],
    )
    
    model_train = dsl.ContainerSpec(
        # name="training pipeline",
        image="nevret/nevret-iris-training:0.5",
        args=[
            '--data', '../01-data-loading/iris.csv'
        ]
    )
    
    model_train.after(data_preprocess)
    
    
    
if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(iris_pipeline, __file__ + ".tar.gz")
    