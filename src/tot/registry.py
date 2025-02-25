MODEL_REGISTRY = {}
STATE_EVALUATOR_REGISTRY = {}
SUCCESSOR_GENERATOR_REGISTRY = {}
STATE_SELECTOR_REGISTRY = {}
STATE_EVALUATOR_REGISTRY = {}
REGISTRIES = [
    MODEL_REGISTRY,
    STATE_EVALUATOR_REGISTRY,
    SUCCESSOR_GENERATOR_REGISTRY,
    STATE_SELECTOR_REGISTRY,
    STATE_EVALUATOR_REGISTRY
]

def get_param(config: dict, key: str):
    value = config.get(key)
    if value is None:
        raise ValueError(f"Key {key} not found in config")
    return value

def register(cls: type, registry: dict):
    registry[cls.__name__] = cls

def get_registry(cls_name: str, registry: dict|None = None):
    if registry is None:
        for reg in REGISTRIES:
            cls = reg.get(cls_name)
            if not cls is None:
                return cls
        raise ValueError(f"Class {cls_name} not found in any registry")
    else:
        cls = registry.get(cls_name)
        if not cls is None:
            return cls
        raise ValueError(f"Class {cls_name} not found in registry")