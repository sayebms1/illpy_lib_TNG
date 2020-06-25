#!/bin/bash
#    run with `sbatch run.sh`

#SBATCH --mail-user=lzkelley@northwestern.edu
###SBATCH -n 1
#SBATCH -n 32
###SBATCH --ntasks-per-node=32

#SBATCH -p hernquist,itc_cluster
#SBATCH --mem-per-cpu=10000
#SBATCH --time=20:00:00
#SBATCH -o match_out.%j
#SBATCH -e match_err.%j
#SBATCH -J match

# python illpy_lib/illbh/details.py --RECREATE=True
# python illpy_lib/illbh/mergers.py --RECREATE=True
# python illpy_lib/illbh/snapshots.py

mpirun -n 32 python illpy_lib/illbh/matcher.py --RECREATE=True
