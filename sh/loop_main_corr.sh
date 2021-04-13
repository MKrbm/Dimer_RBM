#! /bin/bash 




# source activate v_3
WORKDIR=/home/murota/Research/DimerMaster/sh
cd $WORKDIR





# q=(0.9 1 1.1)
# h=(1)
# upperlim_h=${#h[@]}
# upperlim_q=${#q[@]}


# for ((j=0; j<=upperlim_q-1; j++)); do
#     for ((i=0; i<=upperlim_h-1; i++)); do
#         sbatch --job-name=c_h${h[$i]}_q${q[$j]} --output=logfile/corr/h${h[$i]}_q${q[$j]}.log main_corr.sh ${h[$i]} ${q[$j]}
#     done
# done

q=(0.9)
h=(1)
n=($(seq 0 1 9))
upperlim_h=${#h[@]}
upperlim_q=${#q[@]}
upperlim_n=${#n[@]}

for ((l=0; l<=upperlim_n-1; l++)); do
    for ((j=0; j<=upperlim_q-1; j++)); do
        for ((i=0; i<=upperlim_h-1; i++)); do
            sbatch --job-name=c_h${h[$i]}_q${q[$j]}_n${n[$l]} --output=logfile/corr/h${h[$i]}_q${q[$j]}_n${n[$l]}.log main_corr.sh ${h[$i]} ${q[$j]} ${n[$l]}
        done
    done
done

