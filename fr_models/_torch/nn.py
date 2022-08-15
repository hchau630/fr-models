from fr_models import optimize as optim

def state_dict(model):
    state_dict = model.state_dict()
    for name, param in model.named_parameters():
        if isinstance(param, optim.Parameter) and param.custom_name is not None:
            state_dict[param.custom_name] = state_dict[name]
            del state_dict[name]
            
    return state_dict

def load_state_dict(model, state_dict, **kwargs):
    new_state_dict = {}
    for name, param in model.named_parameters():
        if isinstance(param, optim.Parameter) and param.custom_name is not None and param.custom_name in state_dict:
            new_state_dict[name] = state_dict[param.custom_name]
            
    return model.load_state_dict(new_state_dict, **kwargs)
    
def params_dict(model):
    params_dict = {}
    for name, param in model.named_parameters():
        if isinstance(param, optim.Parameter) and param.custom_name is not None:
            params_dict[param.custom_name] = param
        else:
            params_dict[name] = param
            
    return params_dict