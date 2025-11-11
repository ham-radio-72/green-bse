#!/bin/bash 
#SBATCH -p debug
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH -J Nitrogen_bse
#SBATCH -t 0:30:00 
#SBATCH -o solveCasida_output.o%j 

export SCRIPTDIR=/your/code/directory/green-bse/example
export INPUTDIR=/your/input/directory
export SIMDIR=/your/input/directory
export IRDIR=/your/irgrid/directory

python $SCRIPTDIR/solveCasida_main.py --type singlet \
       --beta 1000 --calc_pi 1 --qpac 1 --tda 0 --iter -1 --iter_W -1 --monitor \
       --sim $SIMDIR/sim.h5 --int_path $INPUTDIR/df_hf_int/ \
       --input $INPUTDIR/input.h5 --output $SIMDIR/bseCasida_G2p_clean.h5 --ir_file $IRDIR/1e5_136.h5 \
