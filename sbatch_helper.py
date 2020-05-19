import os
# import sys

# filename    = sys.argv[1]
# script_name = sys.argv[2]
# job_name    = sys.argv[3]
# args        = sys.argv[4:]
filename    = input('Enter filename: ')
script_name = input('Enter name of script to run: ')
job_name    = input('Enter name of job: ')
args        = input('Enter any args to supply to the script, seperated by spaces: ')

# args_str = ''
# for arg in args:
#     args += str(arg) + ' '

f = open(filename, 'w')
f.write('#!/bin/bash\n')
f.write('#\n')
f.write(f'#SBATCH --job-name={job_name}\n')
f.write('#\n')
f.write('#SBATCH --time=24:00:00\n')
f.write('#\n')
f.write('#SBATCH --partition=jamesz,owners,normal\n')
f.write('#SBATCH --ntasks=1\n')
f.write('#SBATCH --cpus-per-task=1\n')
f.write('#SBATCH --mem-per-cpu=10G\n')
f.write(f'srun python {script_name} {args}\n')
f.write('\n')
f.close()