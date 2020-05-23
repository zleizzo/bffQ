import os

new_methods = ['ds', 'bff', 'pd']
opt_methods = ['sgd', 'adam']
lr_decays   = ['f', 'd']
lr_choices  = range(3)

count = 0
params = [None] * 36
for new_method in new_methods:
    for opt_method in opt_methods:
        for lr_decay in lr_decays:
            for lr_choice in lr_choices:
                # for n in ns:
                if new_method == 'pd' and opt_method == 'adam':
                    continue
                param = f'{new_method} {opt_method} {lr_decay} {lr_choice}'
                params[count] = param
                print(param)
                count += 1


for i in range(30):
    filename = f'{i}.sh'
    job_name = f'cp_{i}'
    args     = params[i]
    
    f = open(filename, 'w')
    f.write('#!/bin/bash\n')
    f.write('#\n')
    f.write(f'#SBATCH --job-name={job_name}\n')
    f.write('#\n')
    f.write('#SBATCH --time=10:00:00\n')
    f.write('#\n')
    f.write('#SBATCH --partition=jamesz,owners,normal\n')
    f.write('#SBATCH --ntasks=1\n')
    f.write('#SBATCH --cpus-per-task=1\n')
    f.write('#SBATCH --mem-per-cpu=8G\n')
    f.write('\n')
    f.write(f'srun python final_cartpole.py {args}')
    f.close()