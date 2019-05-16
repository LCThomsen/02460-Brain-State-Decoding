#!/bin/sh
#PBS -N BigRun_mean
#PBS -q hpc
#PBS -l walltime=03:00:00
#PBS -l nodes=1:ppn=4
#PBS -l mem=16gb
#PBS -l vmem=16gb
#PBS -l pmem=8gb

#PBS -m abe


set -e

# THE FOLLOWING SNIPPET SHOULD BE IMPLEMENTED BY FOLLOWING THE</p>
# DIRECTIONS AT https://github.com/AndreasMadsen/my-setup/tree/master/dtu-hpc-python3


module load python3
module load gcc/4.9.2
module load qt
export PYTHONPATH=
source ~/stdpy3/bin/activate

#Running code
#python3 ~/Documents/BigBaseLine.py
#python3 ~/Documents/BigBaseLine_2.py
#python3  ~/Documents/BigBaseLine_cosine.py
#python3  ~/Documents/BigBaseLine_mean.py
python3  ~/Documents/BigBaseLine_ch_mean.py
