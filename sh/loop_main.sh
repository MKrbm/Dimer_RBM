#! /bin/bash 




# source activate v_3
WORKDIR=/home/murota/Research/DimerMaster/sh
cd $WORKDIR





q=(0.9 1 1.1)
h=(1)
upperlim_h=${#h[@]}
upperlim_q=${#q[@]}


for ((j=0; j<=upperlim_q-1; j++)); do
    for ((i=0; i<=upperlim_h-1; i++)); do
        sbatch --job-name=h${h[$i]}_q${q[$j]} --output=logfile/RBM/h${h[$i]}_q${q[$j]}.log main_RBM.sh ${h[$i]} ${q[$j]}
    done
done