import numpy as np
from tot.registry import register, STATE_SELECTOR_REGISTRY, RegistredClass

class StateSelector(RegistredClass):
    def __init__(self, num_states: int):
        self.num_states = num_states
    
    def __call__(self, values:list):
        raise NotImplementedError("select method must be implemented in derived classes")

class SampleStateSelector(StateSelector):
    
    def __call__(self, values:list):
        if sum(values) == 0: 
            # if all values are zero, select randomly
            ps = np.ones(len(values)) / len(values)
        else:
            ps = np.array(values) / sum(values)
        select_ids = np.random.choice(list(range(len(values))), size=self.num_states, p=ps).tolist()
        return select_ids

register(STATE_SELECTOR_REGISTRY, SampleStateSelector, 'sample')

class GreedyStateSelector(StateSelector):
    def __call__(self, values:list):
        select_ids = sorted(list(range(len(values))), key=lambda x: values[x], reverse=True)[:self.num_states]
        return select_ids
    
register(STATE_SELECTOR_REGISTRY, SampleStateSelector, 'greedy')