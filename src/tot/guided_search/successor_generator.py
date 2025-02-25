from tot.tasks.task import Task
from tot.models.models import Model
from tot.registry import register, SUCCESSOR_GENERATOR_REGISTRY, get_param

class SuccessorGenerator:
    def __init__(self, task: Task, model : Model, x: str):
        self.task = task
        self.x = x
        self.model = model
    
    @classmethod
    def from_config(cls, config:dict):
        task = get_param(config, 'task')
        model = get_param(config, 'model')
        x = get_param(config, 'x')
        return cls(task, model, x)

    def generate_successors(self, y: str):
        raise NotImplementedError
    
class SampleSuccessorGenerator(SuccessorGenerator):
    def __init__(self, task: Task, model: Model, x: str, number_of_samples: int):
        super().__init__(task, model, x)
        self.number_of_samples = number_of_samples

    def samples_wrapper(self, y: str):
        raise NotImplementedError

    def generate_successors(self, y: str):
        prompt = self.samples_wrapper(y)
        samples = self.model.generate(prompt, n=self.number_of_samples)
        return [y + s for s in samples]
    
    @classmethod
    def from_config(cls, config:dict):
        task = get_param(config, 'task')
        model = get_param(config, 'model')
        x = get_param(config, 'x')
        number_of_samples = get_param(config, 'number_of_samples')
        return cls(task, model, x, number_of_samples)
    
class StandardSampleSuccessorGenerator(SampleSuccessorGenerator):
    def __init__(self, task: Task, model: Model, x: str, number_of_samples: int):
        super().__init__(task, model, x, number_of_samples)

    def samples_wrapper(self, y: str):
        return self.task.standard_prompt_wrap(self.x, y)

register(StandardSampleSuccessorGenerator, SUCCESSOR_GENERATOR_REGISTRY)

class CotSampleSuccessorGenerator(SampleSuccessorGenerator):
    def __init__(self, task: Task, model: Model, x: str, number_of_samples: int):
        super().__init__(task, model, x, number_of_samples)

    def samples_wrapper(self, y: str):
        return self.task.cot_prompt_wrap(self.x, y)

register(CotSampleSuccessorGenerator, SUCCESSOR_GENERATOR_REGISTRY)

class ProposeSuccessorGenerator(SuccessorGenerator):
    def __init__(self, task: Task, model: Model, x: str):
        super().__init__(task, model, x)
    
    def generate_successors(self, y: str):
        propose_prompt = self.task.propose_prompt_wrap(self.x, y)
        proposals = self.model.generate(propose_prompt, n=1)[0].split('\n')
        return [y + s + '\n' for s in proposals]

register(ProposeSuccessorGenerator, SUCCESSOR_GENERATOR_REGISTRY)