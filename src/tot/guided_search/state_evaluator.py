from tot.tasks.task import Task
from tot.models.model import Model
from tot.registry import register, STATE_EVALUATOR_REGISTRY, RegistredClass

class StateEvaluator(RegistredClass):
    def __init__(self, task:Task):
        self.task = task

    def __call__(self, x:str, ys:list[str]):
        raise NotImplementedError("evaluate method must be implemented in derived classes")

class ModelBasedStateEvaluator(StateEvaluator):
    def __init__(self, task:Task, model: Model, n_evaluate_sample:int):
        super().__init__(task=task)
        self.model = model
        self.n_evaluate_sample = n_evaluate_sample
    
class VoteModelBasedStateEvaluator(ModelBasedStateEvaluator):
    def __call__(self, x:str, ys:list[str]):
        vote_prompt = self.task.vote_prompt_wrap(x, ys)
        vote_outputs = self.model(vote_prompt, n=self.n_evaluate_sample)
        values = self.task.vote_outputs_unwrap(vote_outputs, len(ys))
        return values

register(STATE_EVALUATOR_REGISTRY, VoteModelBasedStateEvaluator, 'vote')

class ValueModelBasedStateEvaluator(ModelBasedStateEvaluator):
    def __init__(self, task:Task, model: Model, n_evaluate_sample:int):
        super().__init__(task, model, n_evaluate_sample)
        self.value_cache = {}

    def evaluate(self, x:str, y:str):
        value_prompt = self.task.value_prompt_wrap(x, y)
        if value_prompt in self.value_cache:
            return self.value_cache[value_prompt]
        value_outputs = self.model(value_prompt, n=self.n_evaluate_sample)
        value = self.task.value_outputs_unwrap(x, y, value_outputs)
        self.value_cache[value_prompt] = value
        return value

    def __call__(self, x:str, ys:list[str]):
        values = []
        local_value_cache = {}
        for y in ys:
            if y in local_value_cache:
                value = 0
            else:    
                value = self.evaluate(x, y)
                local_value_cache[y] = value
            values.append(value)
        return values

register(STATE_EVALUATOR_REGISTRY, ValueModelBasedStateEvaluator, 'value')