#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gpus=v100-32:2


set -x

module load anaconda3/2024.10-1 
conda activate seer

# ==== CONFIGURE PATHS ====
save_checkpoint_path="/ocean/projects/cis250181p/venkatas/seer_libero_pretrain_depth/checkpoints"
root_dir="/ocean/projects/cis250181p/venkatas/Seer/data/LIBERO"
libero_path="/ocean/projects/cis250181p/venkatas/Seer/LIBERO"

mkdir -p ${save_checkpoint_path}

GPUS_PER_NODE=2
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))


srun --ntasks=${WORLD_SIZE} \
     --ntasks-per-node=${GPUS_PER_NODE} \
      torchrun --nproc_per_node=2 --standalone \
       train.py \
         --traj_cons \
         --rgb_pad 10 \
         --gripper_pad 4 \
         --gradient_accumulation_steps 8 \
         --workers 16 \
         --lr_scheduler cosine \
         --save_every_iter 100000 \
         --num_epochs 40 \
         --seed 42 \
         --batch_size 64 \
         --precision fp32 \
         --learning_rate 1e-4 \
         --save_checkpoint \
         --finetune_type libero_pretrain \
         --root_dir ${root_dir} \
         --wandb_project seer \
         --weight_decay 1e-4 \
         --num_resampler_query 3 \
         --run_name libero_pretrain_depth_clip_resnet_action_only \
         --save_checkpoint_path ${save_checkpoint_path} \
         --transformer_layers 6 \
         --transformer_heads 6 \
         --hidden_dim 192 \
         --phase "pretrain" \
         --sequence_length 11 \
         --action_pred_steps 3 \
         --future_steps 3 \
         --atten_goal 4 \
         --window_size 11 \
         --loss_action \
         --gripper_width \
         --atten_only_obs \
         --atten_goal_state \
         --mask_l_obs_ratio 0.5 \
         --warmup_epochs 1 \
         --libero_path ${libero_path} \
         --report_to_wandb \
         --use_text \
         --use_state \
         --use_depth \
         --model_size tiny \
         --seer_mini
