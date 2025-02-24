import itertools
import numpy as np
from functools import partial
from tot.models.models import Model

def bfs(model, args, task, idx, to_print=True):
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [model.get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [model.get_proposals(task, x, y) for y in ys]
        else:
            raise ValueError(f'method_generate {args.method_generate} not recognized')
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = model.get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = model.get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(model, args, task, idx, to_print=True):
    global qwen
    qwen = partial(qwen, temperature=args.temperature)
    print(qwen)
    x = task.get_input(idx)  # input
    ys = model.get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}