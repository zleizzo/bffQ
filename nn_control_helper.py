for i in range(10):
    filename = f'control_{i+1}.sh'
    job_name = f'control_{i+1}'
    
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
    f.write(f'srun python nn_control_pd.py {i + 1}')
    f.close()