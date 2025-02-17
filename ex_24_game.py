import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
from tot.models import get_model

args = argparse.Namespace(backend="Qwen2.5-0.5B-Instruct", temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

model = get_model(args)
task = Game24Task()
ys, infos = solve(model, args, task, 900)
print(ys[0])