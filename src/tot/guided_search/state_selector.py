import numpy as np
from tot.registry import register, STATE_SELECTOR_REGISTRY

class StateSelector:
    def __init__(self, num_states: int):
        self.num_states = num_states
    
    def select(self, values:list):
        raise NotImplementedError("select method must be implemented in derived classes")

class SampleStateSelector(StateSelector):
    def __init__(self, num_states: int):
        super().__init__(num_states)
    
    def select(self, values:list):
        ps = np.array(values) / sum(values)
        select_ids = np.random.choice(list(range(len(values))), size=self.num_states, p=ps).tolist()
        return select_ids

register(SampleStateSelector, STATE_SELECTOR_REGISTRY)

class GreedyStateSelector(StateSelector):
    def __init__(self, num_states: int):
        super().__init__(num_states)
    
    def select(self, values:list):
        select_ids = sorted(list(range(len(values))), key=lambda x: values[x], reverse=True)[:self.num_states]
        return select_ids
    
register(GreedyStateSelector, STATE_SELECTOR_REGISTRY)