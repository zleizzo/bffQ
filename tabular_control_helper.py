methods = ['ub', 'ds', 'bff']
ns = range(5)

for method in methods:
    for n in range(1, 6):
        method_copy = method
        
        if method == 'bff':
            method_copy = str(n) + 'bff'
        print(method_copy)
            
        filename = f'tc_{method_copy}.sh'
        job_name = f'tc_{method_copy}'
        
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
        if method == 'bff':
            f.write(f'srun python tabular_control.py {method} {n}')
        else:
            f.write(f'srun python tabular_control.py {method}')
        f.close()