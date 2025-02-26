MODEL_REGISTRY = {}
STATE_EVALUATOR_REGISTRY = {}
SUCCESSOR_GENERATOR_REGISTRY = {}
STATE_SELECTOR_REGISTRY = {}

REGISTRIES_NAMES = {
    "model" : MODEL_REGISTRY,
    "state_evaluator" : STATE_EVALUATOR_REGISTRY,
    "successor_generator" : SUCCESSOR_GENERATOR_REGISTRY,
    "state_selector" : STATE_SELECTOR_REGISTRY
}

REGISTRIES = [r for r in REGISTRIES_NAMES.values()]

def register( registry: dict, cls: type, cls_name:str|None = None) -> None:
    c_name = cls_name if not cls_name is None else cls.__name__.lower()
    registry[c_name] = cls

def get_registred_class_by_registry(cls_name: str, registry: dict|None) -> type:
    assert registry in REGISTRIES, f"Registry {registry} not found"
    return registry[cls_name]

def get_registred_class_by_registry_name(cls_name:str, registry_name:str|None) -> type:
    assert registry_name in REGISTRIES_NAMES, f"Registry {registry_name} not found"
    registry = REGISTRIES_NAMES[registry_name]
    return get_registred_class_by_registry(cls_name, registry)

def get_registred_class(cls_name: str, registry: dict|None = None, registry_name: str|None=None) -> type:
    if registry is None and registry_name is None:
        raise ValueError("Either registry or registry_name must be provided")
    if registry is not None:
        return get_registred_class_by_registry(cls_name, registry)
    else:
        return get_registred_class_by_registry_name(cls_name, registry_name)

import inspect

class RegistredClass:
    @classmethod
    def from_config(cls, config: dict) -> type:
        signature = inspect.signature(cls.__init__)
        valid_params = list(signature.parameters.keys())[1:]
        for param in valid_params:
            if signature.parameters[param].default is inspect.Parameter.empty and param not in config:
                raise ValueError(f"Missing required parameter: {param}")
        kwargs = {param: config[param] for param in valid_params if param in config}
        
        return cls(**kwargs)
    
def get_registry(class_name: str, config:dict, registry: dict|None = None, registry_name: str|None=None) -> type:
    cls = get_registred_class(class_name, registry, registry_name)
    return cls.from_config(config)

def get_choices(registry: dict|str) -> list[str]:
    reg = registry if isinstance(registry, dict) else REGISTRIES_NAMES[registry]
    return list(reg.keys())