import json
from importlib import import_module
from typing import Dict

def load_model(model_config: Dict):
    model_name = model_config["name"]
    errors = []

    try:
        module_path = f"models.{model_name.lower()}"
        module = import_module(module_path)
        model_class = getattr(module, model_name)
        
        # Extraer y separar la configuración de poda
        model_args = model_config["args"].copy()  # Copia de los argumentos del modelo
        #pruning_params = model_args.pop("pruning", None)  # Extrae "pruning" si existe
        
        # Crear instancia del modelo con los argumentos restantes
        model = model_class(**model_args)
        
        # Aplicar poda si está configurada y habilitada
        #if pruning_params and pruning_params.get("enabled", False):
        #    print(f"Applying pruning with {pruning_params['amount']} on {pruning_params['layers']} layers.")
        #    model.apply_pruning(amount=pruning_params["amount"], layers=pruning_params["layers"])
        
        return model
            
    except ImportError as e:
        errors.append(str(e))
    
    error_messages = '\n'.join(errors)
    raise ValueError(
        f"Failed to import {model_name}. Errors encountered: \n {error_messages}"
    )

def load_config(config_path: str):
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'.")
    except json.JSONDecodeError:
        raise ValueError(f"The file at '{config_path} is not a valid JSON.")
    
    model_config = config.get("model", {}).get("args", {})
    dataset_config = config.get("dataset", {})

    # Sanity check for img_channels and img_size
    if model_config.get("in_channels") != dataset_config.get("img_channels"):
        raise ValueError(
            "Mismatch in 'img_channels' between model and dataset configurations."
        )
    #if model_config.get("img_size") != dataset_config.get("img_size"):
    #    raise ValueError(
    #        "Mismatch in 'img_size' between model and dataset configurations."
    #    )

    return config


    
