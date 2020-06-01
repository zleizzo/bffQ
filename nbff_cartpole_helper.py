for i in range(5):
    for exp_n in range(10):
        filename = f'{exp_n}{i+1}bcart.sh'
        job_name = f'{exp_n}{i+1}bcart'
        
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
        f.write(f'srun python nbff_cartpole.py {i + 1} {exp_n}')
        f.close()