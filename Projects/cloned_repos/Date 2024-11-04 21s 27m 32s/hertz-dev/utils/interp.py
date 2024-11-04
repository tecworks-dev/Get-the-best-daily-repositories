import torch as T
import os 

def rank0():
    rank = os.environ.get('RANK')
    if rank is None or rank == '0':
        return True
    else:
        return False

def print_colored(message, color='reset', bold=False, **kwargs):
    color_dict = {
        'bold': '\033[1m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'grey': '\033[90m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    
    color_code = color_dict.get(color.lower(), color_dict['reset'])
    prefix = color_dict['bold'] if bold else ''
    print(f"{prefix}{color_code}{message}{color_dict['reset']}", **kwargs)

def print0_colored(*args, **kwargs):
    if rank0():
        print_colored(*args, **kwargs)

def param_count(module):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(module)
    output = [f'Total model parameters: {total_params:,}', '---------------------------']
    
    for name, child in module.named_children():
        params = count_parameters(child)
        output.append(f'{name} parameters: {params:,}')
    
    return '\n'.join(output)

def model_size_estimation(module):
    def estimate_size(model):
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return param_size + buffer_size

    total_size = estimate_size(module)
    output = [f'Total model size: {total_size / 1024**2:.2f} MB', '---------------------------']

    for name, child in module.named_children():
        child_size = estimate_size(child)
        output.append(f'{name} size: {child_size / 1024**2:.2f} MB')

    return '\n'.join(output)

def layer_param_distribution(module):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_layer_types(model):
        layer_types = {}
        for name, module in model.named_modules():
            layer_type = module.__class__.__name__
            params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            if params > 0:
                if layer_type not in layer_types:
                    layer_types[layer_type] = 0
                layer_types[layer_type] += params
        return layer_types

    total_params = count_parameters(module)
    layer_types = get_layer_types(module)
    
    output = [f'Total trainable parameters: {total_params:,}', '---------------------------']
    
    for layer_type, count in sorted(layer_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_params) * 100
        output.append(f'{layer_type}: {count:,} ({percentage:.2f}%)')

    return '\n'.join(output)

