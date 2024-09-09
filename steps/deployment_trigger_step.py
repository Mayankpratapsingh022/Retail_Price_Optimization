from zenml import step 


@step 
def deployment_trigger(accuracy: float,min_accuracy: float = 0.9) -> bool:
    return accuracy >min_accuracy