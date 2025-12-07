import os
import requests
import numpy as np
import chex
from ml_collections import config_dict
from disco_rl import agent as agent_lib

# Constants
DEFAULT_AXIS_NAME = 'i'

def unflatten_params(flat_params: chex.ArrayTree) -> chex.ArrayTree:
    """
    Reconstructs a nested parameter dictionary from a flattened one.
    Helper function for load_disco_weights.
    """
    params = {}
    for key_wb in flat_params:
        key = '/'.join(key_wb.split('/')[:-1])
        params[key] = {
            'b': flat_params[f'{key}/b'],
            'w': flat_params[f'{key}/w'],
        }
    return params

def load_disco_weights(
    fname: str = 'disco_103.npz',
    base_url: str = "https://raw.githubusercontent.com/google-deepmind/disco_rl/main/disco_rl/update_rules/weights/"
) -> chex.ArrayTree:
    """
    Downloads the Disco weights if they do not exist locally, loads them,
    and unflattens the parameters.
    """
    if not os.path.exists(fname):
        print(f"Downloading {fname} from {base_url}...")
        url = f"{base_url}{fname}"
        response = requests.get(url)
        response.raise_for_status()
        with open(fname, 'wb') as f:
            f.write(response.content)
    
    print(f"Loading weights from {fname}...")
    with open(fname, 'rb') as file:
        flat_params = np.load(file)
        disco_params = unflatten_params(flat_params)
        
    return disco_params

def get_default_agent_config() -> config_dict.ConfigDict:
    """
    Returns the default agent configuration based on meta_train.ipynb settings.
    """
    # Create settings for an agent.
    agent_settings = agent_lib.get_settings_disco()
    agent_settings.net_settings.name = 'mlp'
    agent_settings.net_settings.net_args = dict(
        dense=(512, 512),
        model_arch_name='lstm',
        head_w_init_std=1e-2,
        model_kwargs=dict(
            head_mlp_hiddens=(256,),
            lstm_size=256,
        ),
    )
    agent_settings.learning_rate = 5e-4
    
    return agent_settings