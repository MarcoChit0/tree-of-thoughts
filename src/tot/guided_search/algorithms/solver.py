import itertools
import numpy as np
from functools import partial
from tot.models.models import Model
from tot.tasks.task import Task

class Solver:
    def __init__(self, model : Model, task : Task, config):
        self.model = model
        self.task = task
        self.config = config
    
    def solve(self, idx):
        raise NotImplementedError

class BFSSolver(Solver):
    def __init__(self, model, task, config):
        super().__init__(model, task, config)   
    
    def solve(self, idx):
        x = self.task.get_input(idx)
        ys = ['']
        infos = []
        for step in range(self.task.steps):
            if self.config.method_generate == 'sample':
                new_ys = [self.model.get_samples(self.task, x, y, self.config.n_generate_sample, prompt_sample=self.config.prompt_sample) for y in ys]
            elif self.config.method_generate == 'propose':
                new_ys = [self.model.get_proposals(self.task, x, y) for y in ys]
            else:
                raise ValueError(f'method_generate {self.config.method_generate} not recognized')
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            if self.config.method_evaluate == 'vote':
                values = self.model.get_votes(self.task, x, new_ys, self.config.n_evaluate_sample)
            elif self.config.method_evaluate == 'value':
                values = self.model.get_values(self.task, x, new_ys, self.config.n_evaluate_sample)
            if self.config.method_select == 'sample':
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(ids, size=self.config.n_select_sample, p=ps).tolist()
            elif self.config.method_select == 'greedy':
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:self.config.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]
            infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
            ys = select_new_ys
        return ys, {'steps': infos}

class NaiveSolver(Solver):
    def __init__(self, model, task, config):
        super().__init__(model, task, config)
    
    def solve(self, idx):
        x = self.task.get_input(idx)
        ys = self.model.get_samples(self.task, x, '', self.config.n_generate_sample, self.config.prompt_sample, stop=None)
        return ys, {}