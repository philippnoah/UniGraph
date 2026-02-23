#!/bin/bash
#SBATCH --job-name=unigraph
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home1/eibl/gfm/UniGraph/logs/%j.out
#SBATCH --error=/home1/eibl/gfm/UniGraph/logs/%j.err

cd $SLURM_SUBMIT_DIR
mkdir -p $SLURM_SUBMIT_DIR/logs

module load cuda/12.4

source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda activate unigraph

cd $SLURM_SUBMIT_DIR

CUDA_VISIBLE_DEVICES=0 python main.py \
  --run_entity eibl-usc \
  --device 0 \
  --seed 0 \
  --mask_rate 0.5 \
  --dropout 0.0 \
  --drop_edge_rate 0.0 \
  --num_layers 2 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --lr_f 1e-2 \
  --cut_off 96 \
  --batch_size 48 \
  --eval_batch_size 256 \
  --num_epochs 3 \
  --num_roots 10 \
  --length 8 \
  --gnn_type gat \
  --lp_epochs 100 \
  --eval_steps 1 \
  --eval_interval 3000 \
  --save_steps 1 \
  --hidden_size 768 \
  --lm_type microsoft/deberta-base \
  --datasets_name FB15K237 \
  --eval_datasets_name FB15K237 \
  --process_mode TA \
  --lam 0.1
