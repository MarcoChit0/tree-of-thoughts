from tot.tasks.task import Task
from tot.models.model import Model
from tot.registry import register, SUCCESSOR_GENERATOR_REGISTRY, RegistredClass

class SuccessorGenerator(RegistredClass):
    def __init__(self, task: Task, model : Model):
        self.task = task
        self.model = model

    def __call__(self, x:str, ys: list[str]):
        return [self.generate_successor(x, y) for y in ys]

    def generate_successor(self, x:str, y:str):
        raise NotImplementedError
    
class SampleSuccessorGenerator(SuccessorGenerator):
    def __init__(self, task: Task, model: Model, number_of_samples: int):
        super().__init__(task, model)
        self.number_of_samples = number_of_samples

    def samples_wrapper(self, x:str, y: str):
        raise NotImplementedError

    def generate_successor(self, x:str, y:str):
        prompt = self.samples_wrapper(x, y)
        samples = self.model(prompt, n=self.number_of_samples)
        return [y + s for s in samples]
    
class StandardSampleSuccessorGenerator(SampleSuccessorGenerator):
    def samples_wrapper(self, x: str, y: str):
        return self.task.standard_prompt_wrap(x, y)

register(SUCCESSOR_GENERATOR_REGISTRY, StandardSampleSuccessorGenerator, 'standard-sample')

class CotSampleSuccessorGenerator(SampleSuccessorGenerator):
    def samples_wrapper(self, x: str, y: str):
        return self.task.cot_prompt_wrap(x, y)

register(SUCCESSOR_GENERATOR_REGISTRY, CotSampleSuccessorGenerator, 'cot-sample')
    

class ProposeSuccessorGenerator(SuccessorGenerator):
    def generate_successor(self, x:str, y:str):
        propose_prompt = self.task.propose_prompt_wrap(x, y)
        proposals = self.model(propose_prompt, n=1)[0].split('\n')
        return [y + s + '\n' for s in proposals]
    

register(SUCCESSOR_GENERATOR_REGISTRY, ProposeSuccessorGenerator, 'propose')