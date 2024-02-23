from kfp import dsl
import kfp.components as comp

from logistic_regression.logistic_regression import *

@dsl.pipeline(
    name='test',
    description='test'
)
def test_pipeline():
    logistic_regression_task = logistic_regression()
    return logistic_regression_task.output
    

    
    
if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(test_pipeline, __file__ + "pipeline.yaml")
    