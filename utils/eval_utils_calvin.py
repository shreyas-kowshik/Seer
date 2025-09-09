from collections import defaultdict, namedtuple
import logging
import os, json, random
from pathlib import Path
import sys
import time
import PIL.Image as Image
import copy
from collections import deque
from moviepy.editor import ImageSequenceClip
import imageio
import matplotlib.pyplot as plt

sys.path.append('/home/skowshik/work/calvin/calvin_models')
from calvin_agent.models.calvin_base_model import CalvinBaseModel
import time
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
import torch
from tqdm.auto import tqdm
from calvin_env.envs.play_table_env import get_env
from utils.data_utils import preprocess_image, preprocess_text_calvin
import functools
from utils.train_utils import get_cast_dtype
import cv2

import torch.distributed as dist


os.environ['PYOPENGL_PLATFORM'] = 'egl'
logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000
RETURN_IMAGE_PRED_MODEL = True

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    return env

def unpatchify(x, patch_size=16):
    """
    x: (N, L, patch_size**2 * 3)
    imgs: (N, 3, H, W)
    """
    # Calculate h and w from the number of patches (L)
    h = w = int(x.shape[1]**0.5)
    
    # Reshape to (N, h, w, patch_size, patch_size, 3)
    x = x.reshape(x.shape[0], h, w, patch_size, patch_size, 3)
    
    # Rearrange the dimensions to (N, 3, h, patch_size, w, patch_size)
    x = np.einsum('nhwpqc->nchpwq', x)
    
    # Reshape to the image format (N, 3, H, W)
    imgs = x.reshape(x.shape[0], 3, h * patch_size, w * patch_size)
    
    return imgs

def image_pred_tensor2img(x, img_id=0, seq_id=-1):
    """
    Converts the predicted image tensor to an actual image for visualization

    x: (seq_len, num_images, patch_size**2 * 3)
    img_id: Which of `num_images` to convert
    seq_id: Which of `seq_len` to convert
    """
    # Code for visualizing predicted image #
    single_prediction = x.cpu()[seq_id, img_id, :, :][np.newaxis, ...] # Take the last timestep
    reconstructed_image = unpatchify(single_prediction)
    image_to_show = reconstructed_image.squeeze(0)
    min_val = np.min(image_to_show)
    max_val = np.max(image_to_show)
    image_to_show = (image_to_show - min_val) / (max_val - min_val)
    image_to_show = image_to_show.transpose(1, 2, 0)
    image_to_show = (image_to_show * 255).astype(np.uint8)
    ######

    return image_to_show

def update_append_dict_with_info_dict(append_dict, info_dict):
    """
    info_dict: A Dictionary object storing keys and values
    append_dict: A Dictionary object that appends value of each key in `info_dict` to the same key 
    """
    for (k, v) in info_dict.items():
        if k not in append_dict:
            append_dict[k] = []
        append_dict[k].append(v)

def flatten(arrays):
    return [item for sublist in arrays for item in sublist]

class ModelWrapper(CalvinBaseModel):
    def __init__(self, model, tokenizer, image_processor, cast_dtype, history_len=10, 
                calvin_eval_max_steps=360, action_pred_steps=3):
        super().__init__()
        self.model = model
        self.cast_type = cast_dtype
        self.use_diff = False
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.history_len = history_len
        self.calvin_eval_max_steps = calvin_eval_max_steps
        self.action_pred_steps = action_pred_steps
        self.device = "cuda"
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.act_queue = deque(maxlen=history_len-1)

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.act_queue = deque(maxlen=self.history_len-1)

    def step(self, obs, goal, timestep, return_pred_img=False):
        out_dict = {} # Stores stuff to store in output

        image = obs["rgb_obs"]['rgb_static']
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_type)

        gripper = obs["rgb_obs"]['rgb_gripper']
        gripper = Image.fromarray(gripper)
        gripper = self.image_process_fn([gripper])
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type)

        text_x = self.text_process_fn([goal])
        text_x = text_x.unsqueeze(1)

        state = obs['robot_obs']
        state = torch.from_numpy(np.stack([state]))
        state = state.unsqueeze(1).to(dtype=self.cast_type)
        state = torch.cat([state[..., :6], state[..., [-1]]], dim=-1)

        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            text_x = text_x.to(device)
            gripper = gripper.to(device)
            state = state.to(device)
            self.img_queue.append(image_x)  
            self.gripper_queue.append(gripper)
            self.state_queue.append(state)
            if len(self.text_queue) == 0 and text_x is not None:  
                self.text_queue.append(text_x)
                for _ in range(self.model.module.sequence_length - 1):
                    self.text_queue.append(text_x)
            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)
            num_step = image_primary.shape[1]
            if num_step < self.history_len:  
                input_image_primary = torch.cat([image_primary, image_primary[:, -1].repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat([image_wrist, image_wrist[:, -1].repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_state = torch.cat([state, state[:, -1].repeat(1, self.history_len-num_step, 1)], dim=1)
            else:
                input_image_primary = image_primary
                input_image_wrist = image_wrist
                input_state = state
            arm_action, gripper_action, image_pred, arm_pred_state, gripper_pred_state, _ = self.model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=torch.zeros(1, self.history_len, 7).to(input_state.device),
            )

            # Convert predicted images for visualization
            image_pred0_1 = image_pred_tensor2img(image_pred, 0, -1)
            image_pred1_1 = image_pred_tensor2img(image_pred, 1, -1)
            image_pred00 = image_pred_tensor2img(image_pred, 0, 0)
            image_pred10 = image_pred_tensor2img(image_pred, 1, 0)
            
            action = torch.concat((arm_action[0, :, 0, :], gripper_action[0, :, 0, :] > 0.5), dim=-1)
            action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
            action = action.cpu().detach().to(dtype=torch.float16).numpy()
            if num_step < self.history_len:
                action = action[num_step - 1]
            else:
                action = action[-1]
            
            # Populate out_dict
            out_dict['action'] = action
            out_dict['image_pred0_1'] = image_pred0_1
            out_dict['image_pred1_1'] = image_pred1_1
            out_dict['image_pred00'] = image_pred00
            out_dict['image_pred10'] = image_pred10

        return out_dict

def evaluate_policy_ddp(model, env, epoch, calvin_conf_path, eval_log_dir=None, debug=False, create_plan_tsne=False, reset=False, diverse_inst=False, custom_eval_sequences=None):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open('./utils/lang_annotation_cache.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    if custom_eval_sequences is None:
        with open('./utils/eval_sequences.json', 'r') as f:
            eval_sequences = json.load(f)
    else:
        with open(custom_eval_sequences, 'r') as f:
            eval_sequences = json.load(f)
    
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    assert NUM_SEQUENCES % device_num == 0
    interval_len = int(NUM_SEQUENCES // device_num)
    eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir, base_sequence_i+local_sequence_i, reset=reset, diverse_inst=diverse_inst, custom_eval_sequences=custom_eval_sequences)
        results.append(result)
        eval_sequences.set_description(
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
        )
        local_sequence_i += 1
    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]
    
    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_list = [_[0] for _ in res_tup_list]
        eval_seq_list = [_[1] for _ in res_tup_list]
        print_and_save(res_list, eval_seq_list, eval_log_dir, epoch)

    return results

def get_gif_folder(custom_eval_sequences, eval_log_dir):
    task_name = custom_eval_sequences.split('/')[-1].split('.')[0]
    # Create a folder to save the GIFs for the custom set of sequences
    gif_folder = os.path.join(eval_log_dir, task_name)
    return gif_folder

def save_gif_from_image_array(images_array, custom_eval_sequences, eval_log_dir, main_name='', append_name='', fps=25):
    task_name = custom_eval_sequences.split('/')[-1].split('.')[0]
    # Create a folder to save the GIFs for the custom set of sequences
    gif_folder = os.path.join(eval_log_dir, task_name)
    os.makedirs(gif_folder, exist_ok=True)
    gif_path = os.path.join(gif_folder, f"{main_name}_{append_name}.gif")
    imageio.mimsave(gif_path, images_array, fps=fps)

def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False, custom_eval_sequences=None, save_success_gifs=True):
    """
    Evaluates a sequence of language instructions.

    eval_sequence: list of subtasks
        ['rotate_blue_block_right', 'move_slider_right', 'lift_red_block_slider', 'place_in_slider', 'turn_off_lightbulb']
    
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    gif_keys = ['image_pred0_1', 'image_pred1_1', 'image_pred00', 'image_pred10', 'env_render_image'] # Create gifs for these keys

    eval_sequence_out_dict = {} # Collates keys for entire `eval_sequence` across subtasks
    for subtask_i, subtask in enumerate(eval_sequence):        
        if reset:
            success, episode_out_dict = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, diverse_inst=diverse_inst, robot_obs=robot_obs, scene_obs=scene_obs, custom_eval_sequences=custom_eval_sequences)
        else:
            success, episode_out_dict = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, diverse_inst=diverse_inst, custom_eval_sequences=custom_eval_sequences)
        
        # Update `out_dict_eval_sequence` with `episode_out_dict` keys and values
        update_append_dict_with_info_dict(eval_sequence_out_dict, episode_out_dict)
        
        if success:
            success_counter += 1
        else:
            # Flatten `eval_sequence_out_dict`
            for k in eval_sequence_out_dict.keys():
                eval_sequence_out_dict[k] = flatten(eval_sequence_out_dict[k])
            # Save GIFs
            for gif_key in gif_keys:
                save_gif_from_image_array(eval_sequence_out_dict[gif_key], custom_eval_sequences, eval_log_dir, main_name="{}_{}_{}".format(sequence_i, subtask, gif_key), append_name='failure', fps=25)

            # Save initial state and eval_sequence to files #
            gif_folder = get_gif_folder(custom_eval_sequences, eval_log_dir)
            json.dump(initial_state, open(os.path.join(gif_folder, "{}_initial_state.json".format(sequence_i)), "w"))
            json.dump(eval_sequence, open(os.path.join(gif_folder, "{}_eval_sequence.json".format(sequence_i)), "w"))

            return success_counter

    # Flatten `eval_sequence_out_dict`
    for k in eval_sequence_out_dict.keys():
        eval_sequence_out_dict[k] = flatten(eval_sequence_out_dict[k])
    # Save GIFs
    for gif_key in gif_keys:
        save_gif_from_image_array(eval_sequence_out_dict[gif_key], custom_eval_sequences, eval_log_dir, main_name="{}_{}_{}".format(sequence_i, subtask, gif_key), append_name='success', fps=25)
    # Save initial state and eval_sequence to files #
    gif_folder = get_gif_folder(custom_eval_sequences, eval_log_dir)
    json.dump(initial_state, open(os.path.join(gif_folder, "{}_initial_state.json".format(sequence_i)), "w"))
    json.dump(eval_sequence, open(os.path.join(gif_folder, "{}_eval_sequence.json".format(sequence_i)), "w"))
    
    return success_counter

def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, eval_log_dir='', subtask_i=-1, sequence_i=-1, robot_obs=None, scene_obs=None, diverse_inst=False, custom_eval_sequences=None):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).

    subtask: Current language subtask
    """
    planned_actions = []
    
    episode_out_dict = {} # Stores tensors for entire episode

    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
    model.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):
        out_dict = model.step(obs, lang_annotation, step)
        action = out_dict['action']
        
        if len(planned_actions) == 0:
            if action.shape == (7,):
                planned_actions.append(action)
            else:
                planned_actions.extend([action[i] for i in range(action.shape[0])])
        action = planned_actions.pop(0)

        if custom_eval_sequences is not None:
            # Get rgb observation of the current step
            rgb = env.render(mode="rgb_array")[:,:,::-1]
            out_dict['env_render_image'] = rgb

        obs, _, _, current_info = env.step(action)

        # Update episode_dict
        update_append_dict_with_info_dict(episode_out_dict, out_dict)
        
        if step == 0:
            collect_plan(model, plans, subtask)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True, episode_out_dict
    
    return False, episode_out_dict

import pdb

def eval_one_epoch_calvin_ddp(args, model, dataset_path, image_processor, tokenizer, eval_log_dir=None, debug=False, future_act_len=-1, reset=False, diverse_inst=False):
    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = args.sequence_length
    wrapped_model = ModelWrapper(
                        model, 
                        tokenizer, 
                        image_processor, 
                        cast_dtype, 
                        history_len=hist_len, 
                        calvin_eval_max_steps=EP_LEN,
                        action_pred_steps = args.action_pred_steps)
    evaluate_policy_ddp(wrapped_model, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, debug=debug, reset=reset, diverse_inst=diverse_inst, custom_eval_sequences=args.custom_eval_sequences)
