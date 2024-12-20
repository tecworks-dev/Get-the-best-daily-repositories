import math
import numpy as np

def norm_pdf(x, mean, sd):
    var = float(sd)**2
    denom = (2 * math.pi * var)**.5
    num = math.exp(-(float(x) - float(mean))**2 / (2 * var))
    return num / denom

def norm_cdf(x, mean, sd):
    # calculate the cumulative distribution function for the normal distribution
    return (1. + math.erf((x - mean) / (math.sqrt(2) * sd))) / 2.

def normalize_params(params, params_dict):
    # 1. normalize the GT params into [-1, 1] range 2. convert discrete params into continuous
    keys_p, keys_d = params.keys(), params_dict.keys()
    assert set(keys_p) == set(keys_d)
    for key in params.keys():
        param_type = params_dict[key][0]
        if param_type == "discrete":
            # assume discrete params are represented as continuous integers
            choices = params_dict[key][1]
            leng = len(choices)
            if choices[0].__class__ == float:
                # TODO: this is to fix float discrete params
                idx = np.where(np.array(choices) == float(params[key]))[0][0]
            else:
                idx = np.where(np.array(choices) == float(int(params[key])))[0][0]
            # uniformly partition the target [-1, 1] range into equal parts
            # set the discrete value as the middle point of the partition
            params[key] = 2.0 / leng * (idx + 0.5) - 1 
        elif param_type == "continuous":
            # uniformly project the parameter value into [-1, 1] range
            min_v, max_v = params_dict[key][1]
            params[key] = ((params[key] - min_v) / (max_v - min_v)) * 2 - 1
        elif param_type == "normal":
            mean, std = params_dict[key][1]
            # clamp the normal distribution to [mean-3\sigma, mean+3\sigma], 
            # and then uniformly project the value into [-1, 1] range
            min_v, max_v = mean - 3 * std, mean + 3 * std
            params[key] = ((params[key] - min_v) / (max_v - min_v)) * 2 - 1
        else:
            raise NotImplementedError
    return params

def unnormalize_params(params, params_dict):
    # project the parameters back to the original range
    # TODO: assume the params is ordered in the same order as params_dict keys
    keys = params_dict.keys()
    params_u = {}
    for i, key in enumerate(keys):
        param_type = params_dict[key][0]
        # do the clamp to the predicted params into [-1, 1]
        params_i = np.clip(params[i], -1, 1)
        if param_type == "discrete":
            choices = params_dict[key][1]
            leng = len(choices)
            idx = (params_i + 1) // (2 / leng)
            if idx > leng - 1:
                idx = leng - 1
            params_u[key] = choices[int(idx)]
        elif param_type == "continuous":
            min_v, max_v = params_dict[key][1]
            params_u[key] = ((params_i + 1) / 2) * (max_v - min_v) + min_v
        elif param_type == "normal":
            mean, std = params_dict[key][1]
            min_v, max_v = mean - 3 * std, mean + 3 * std
            params_u[key] = ((params_i + 1) / 2) * (max_v - min_v) + min_v
        else:
            raise NotImplementedError
    return params_u