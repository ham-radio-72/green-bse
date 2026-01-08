#!/bin/bash 

export SCRIPTDIR=../../script
export SIMDIR=./scgw
export IRDIR=./irgrid

python -u $SCRIPTDIR/solveCasida_main.py \
       --type singlet --beta 1000 \
       --qpac 1 --iter -1 --iter_W -1 \
       --calc_pi 1 --monitor 1 \
       --sim $SIMDIR/sim.h5 \
       --int_path ./df_hf_int/ \
       --ir_file $IRDIR/1e5_136.h5 \
       --input ./input.h5 \
       --output ./bse_singlet.h5 
       
