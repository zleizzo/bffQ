import os

new_methods = ['pd']
opt_methods = ['sgd']
lr_decays   = ['d']
lr_choices  = [0]
ns          = range(10)

count = 0
params = [None] * (10 * 2)
for new_method in new_methods:
    for opt_method in opt_methods:
        for lr_decay in lr_decays:
            for lr_choice in lr_choices:
                for n in ns:
                    if new_method == 'pd' and opt_method == 'adam':
                        continue
                    param = f'{new_method} {opt_method} {lr_decay} {lr_choice} {n}'
                    params[count] = param
                    print(f'{count}: {param}')
                    count += 1


for i in range(len(params)):
    filename = f'cp_{i + 20}.sh'
    job_name = f'cp_{i + 20}'
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