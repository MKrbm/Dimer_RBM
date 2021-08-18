#!/bin/sh
#SBATCH --get-user-env
#SBATCH -p except12
#SBATCH --mem=5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time 24:00:00
#SBATCH --mail-user=mukeisuke0709@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=2


WORKDIR=/home/murota/Research/DimerMaster
cd $WORKDIR
source activate netket_v3

python -u main_corr.py --h $1 --q $2  > sh/output/corr/h$1_q$2L=20.out