#!/bin/sh
#SBATCH --get-user-env
#SBATCH -p except11
#SBATCH --mem=4000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time 24:00:00
#SBATCH --mail-user=mukeisuke0709@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=13


WORKDIR=/home/murota/Research/DimerMaster
cd $WORKDIR
source activate netket_v3

python main_dynamics.py --h $1 --q $2