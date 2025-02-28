import itertools
import numpy as np
from functools import partial
from tot.models.model import Model
from tot.tasks.task import Task
from tot.guided_search.state_evaluator import StateEvaluator
from tot.guided_search.successor_generator import SuccessorGenerator
from tot.guided_search.state_selector import StateSelector

class Solver:
    def __init__(self, task : Task, model : Model, state_selector : StateSelector, state_evaluator : StateEvaluator, successor_generator : SuccessorGenerator):
        self.model = model
        self.task = task
        self.state_selector = state_selector
        self.state_evaluator = state_evaluator
        self.successor_generator = successor_generator
    
    def solve(self, idx) -> tuple[list[str], dict]:
        raise NotImplementedError
    
    def __call__(self, idx) -> tuple[list[str], dict]:
        x = self.task.get_input(idx)
        return self.solve(x)

# TODO: acredito ter errado os tipos em algum momento... ys não faz sentido do jeito que está

class BFSSolver(Solver):
    def solve(self, x:str) -> tuple[list[str], dict]:
        ys = ['']
        infos = []
        print(f"x:\n{x}\n")
        for step in range(self.task.steps):
            print("previous ys:\n")
            for y in ys:
                print(y)
            new_ys = self.successor_generator(x, ys)
            new_ys = list(itertools.chain(*new_ys))
            print('\nnew ys:\n')
            print(new_ys)
            print(type(new_ys))
            for y in new_ys:
                print(y)
            values = self.state_evaluator(x, new_ys)
            print('\nvalues:\n')
            print(values)
            select_ids = self.state_selector(values)
            select_new_ys = [new_ys[select_id] for select_id in select_ids]
            print('\nselect new ys:\n')
            for y in select_new_ys:
                print(y)
            infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
            ys = select_new_ys
            exit(1)
        return ys, {'steps': infos}

class NaiveSolver(Solver):
    def solve(self, x:str) -> tuple[list[str], dict]:
        ys = self.successor_generator_successor(x, y='')
        return ys, {}