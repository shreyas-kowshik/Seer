#!/bin/bash
set -euo pipefail

# Disable Numba JIT compilation to prevent crashes
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# # Enforce EGL in the environment too (redundant with step 1, but helpful for other scripts)
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl
# export TORCH_NCCL_BLOCKING_WAIT=1

# Some env related stuff
cd /ocean/projects/cis250181p/kowshik/LIBERO/
ln -sf libero/libero/bddl_files ./bddl_files
ln -sf libero/libero/init_files ./init_files  
ln -sf libero/libero/assets ./assets
cd /ocean/projects/cis250181p/kowshik/Seer/

# (Optional) quiet old var
unset NCCL_BLOCKING_WAIT
### ===== CONFIGURE THESE PATHS ===== ###
save_checkpoint_path="checkpoints/"
root_dir="/ocean/projects/cis250181p/kowshik/"
libero_path="/ocean/projects/cis250181p/kowshik/LIBERO/"
### ================================== ###
# save_checkpoint_path="checkpoints/"
LOG_DIR="logs"
ckpt_id=0
mkdir -p ${LOG_DIR}
test_id="${ckpt_id}"
logfile="${LOG_DIR}/${test_id}.log"
node=1
node_num=1

# Set additional environment variables for stability
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python -m torch.distributed.run --nnodes=${node} --nproc_per_node=${node_num} --master_port=10212 eval_libero.py\
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --workers 4 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 40 \
    --seed 42 \
    --batch_size 3 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --save_checkpoint \
    --finetune_type "libero_10" \
    --root_dir ${root_dir} \
    --weight_decay 1e-4 \
    --num_resampler_query 3 \
    --run_name libero_eval_depth_clip_resnet_action_only \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 6 \
    --transformer_heads 6 \
    --hidden_dim 192 \
    --phase "evaluate" \
    --sequence_length 11 \
    --action_pred_steps 3 \
    --future_steps 3 \
    --atten_goal 4 \
    --gripper_width \
    --atten_only_obs \
    --use_text \
    --use_state \
    --use_depth \
    --model_size tiny \
    --seer_mini \
    --eval_libero_ensembling \
    --resume_from_checkpoint /ocean/projects/cis250181p/kowshik/Seer/checkpoints/libero_pretrain_depth_clip_resnet_action_only_v2/4.pth\
    --libero_path ${libero_path} \
    --encoder_type resnet \
    --model_size tiny \
    --seer_mini
