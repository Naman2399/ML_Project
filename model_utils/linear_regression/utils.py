def get_model_parameters(model):
    """
    Get the parameters (weights and bias) of a PyTorch model.

    Args:
    - model: PyTorch model

    Returns:
    - weights: Tensor containing the weights of the model
    - bias: Tensor containing the bias of the model
    """
    parameters = model.state_dict()
    weights = parameters['linear.weight']
    bias = parameters['linear.bias']

    print("Weights:", weights)
    print("Bias:", bias)

    return weights, bias