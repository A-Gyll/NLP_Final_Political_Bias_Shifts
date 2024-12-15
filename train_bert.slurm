#!/bin/bash

#Submit this script with: sbatch train_bert.slurm

#SBATCH --time=10:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=1   # number of CPU cores per task
#SBATCH --gres=gpu:a100:1   # gpu devices per node
#SBATCH --constraint=a100_80gb
#SBATCH --partition=gpu   # partition
#SBATCH --mem=32G   # memory
#SBATCH -J "Train Bert"   # job name
#SBATCH --mail-user=ujx4ab@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=grad-nlp-cs6501   # allocation name
#SBATCH -o train_bert-%u-%j.out
#SBATCH -e train_bert-%u-%j.err


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

source env.sh
time python ./src/LLMs/fine_tune_bert_slurm.py -d "data/Sub_samples/CNN_2024-09-02T00:00:00+00:00.csv"