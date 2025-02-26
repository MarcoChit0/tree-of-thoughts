import os
import json
import argparse

from tot.tasks import get_task
from tot.models.hugging_face import *
from tot.models.model import *
from tot.registry import *
from tot.guided_search import *
from tot.guided_search.algorithms.solver import BFSSolver

def run(args):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    if args.naive_run:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    model = get_registry(args.backend, {'temperature': args.temperature, 'backend': args.backend}, registry_name='model')
    state_selector = get_registry(args.method_select, {'num_states': args.n_select_sample}, registry_name='state_selector')
    state_evaluator = get_registry(args.method_evaluate, {'task': task, 'model':model, 'n_evaluate_sample':args.n_evaluate_sample}, registry_name='state_evaluator')
    prompt_sample = f"{args.prompt_sample}-sample" if args.method_generate == "sample" else args.method_generate
    successor_generator = get_registry(prompt_sample, {'task': task, 'model': model, 'number_of_samples': args.n_generate_sample}, registry_name='successor_generator')
    solver = BFSSolver(task, model, state_selector, state_evaluator, successor_generator)
    for i in range(args.task_start_index, args.task_end_index):
        # solve
        ys, info = solver(i)

        # log
        infos = [task.test_output(i, y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': model.get_usage()})
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)
        
        # log main metric
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
    
    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print('usage_so_far', model.get_usage())


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=get_choices("model"), default='Qwen2.5-0.5B-Instruct')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=get_choices("state_evaluator"))
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)