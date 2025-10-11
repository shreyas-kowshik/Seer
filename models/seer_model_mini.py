import os
import random
from functools import partial
from copy import deepcopy
from timm.models.vision_transformer import Block
import torch
import time
from torch import nn
import torch.nn.functional as F
import clip
import numpy as np
from models.vit_mae import MaskedAutoencoderViT, RGBD_CLIP_RoPE_Embedder
from models.perceiver_resampler import PerceiverResampler
from models.gpt2 import GPT2Model
from transformers import GPT2Config
from pdb import set_trace
import random 
import torchvision.models as tv_models

# ---- size presets -----------------------------------------------------------
from dataclasses import dataclass
def generate_attention_mask(K, num_A, num_B, atten_goal, atten_goal_state,
                            atten_only_obs,
                            attn_robot_proprio_state,
                            mask_l_obs_ratio,
                            num_obs_token, action_pred_steps):
    # num_A: 1+1+self.NUM_RESAMPLER_QUERY*2+1*2
    # num_A: text, state, image_embedding, image_cls_token_embedding
    # num_B: self.NUM_OBS_TOKEN+self.action_pred_steps
    # num_B: obs_tokens(if exists), action_pred_token, state_pred_token (if exists)
    sequence_length = (num_A + num_B) * K
    attention_mask = torch.zeros((sequence_length, sequence_length))
    for i in range(K):
        start_index = i * (num_A + num_B)
        end_index = start_index + num_A + num_B
        
        # the i-th sub-sequence can not attend to the sub-sequences that after the i-th
        attention_mask[start_index:end_index, end_index:] = -float('inf')
        
        # the sub-sub-sequence B can not be attended to
        attention_mask[:, start_index+num_A:end_index] = -float('inf')
        
        # if obs_token exists, action_pred_token should attend to it
        if num_obs_token > 0 and action_pred_steps:
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
        if num_obs_token > 0 and atten_only_obs and action_pred_steps:
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps] = -float('inf')
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+2:start_index+num_A] = 0.0
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
            if attn_robot_proprio_state:
                attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+1:start_index+2] = 0.0
            if mask_l_obs_ratio > 0:
                count = int(mask_l_obs_ratio * (num_obs_token))
                selected_numbers = np.random.choice(range(num_obs_token), size=count, replace=False)
                for num in selected_numbers:
                    attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A+num] = -float('inf')
        if num_obs_token > 0 and atten_goal:
            if i < K - atten_goal:
                pred_end_index = (i + atten_goal) * (num_A + num_B)
                if atten_goal_state:
                    attention_mask[start_index+num_A:start_index+num_A+num_obs_token,pred_end_index+1:pred_end_index+2] = 0.0

    return attention_mask

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

@dataclass
class SeerSize:
    name: str
    hidden_dim: int
    transformer_layers: int
    transformer_heads: int
    num_resampler_query: int       # per image (ViT path)
    num_obs_token_per_image: int   # for obs_pred (ViT/ResNet both use this)
    perceiver_depth: int
    dino_variant:str

SEER_PRESETS = {
    "base":  SeerSize("base", 384, 24, 12, 6, 9, 3, "dinov2_vitb14"),
    "small": SeerSize("small", 256, 8,  8,  4, 6, 1, "dinov2_vits14"),
    "tiny":  SeerSize("tiny",  192, 6,  8,  3, 4, 1, "dinov2_vits14"),
}
# ---------------------------------------------------------------------------

class SeerAgentMini(nn.Module):
    def __init__(
        self,
        finetune_type,
        clip_device,
        vit_checkpoint_path,
        sequence_length=10,
        num_resampler_query=9,
        num_obs_token_per_image=10,
        obs_pred=False,
        atten_only_obs=False,
        attn_robot_proprio_state=False,
        atten_goal=False,
        atten_goal_state=False,
        mask_l_obs_ratio=0.0,
        calvin_input_image_size=224,
        patch_size=16,
        mask_ratio=0.0,
        num_token_per_timestep=41,
        input_self=False,
        action_pred_steps=1,
        transformer_layers=12,
        hidden_dim=384,
        transformer_heads=12,
        phase="",
        gripper_width=False,
        model_size: str = None,                  # {"tiny","small","base"} or None to use explicit dims
        encoder_type: str = "vit",               # {"vit","resnet"}
        resnet_variant: str = "resnet18",        # torchvision name
        resnet_pretrained: bool = True,
        allow_obs_pred_with_resnet: bool = True, # you asked this to be a flag
        use_text: bool = True,
        use_state: bool = True,
        use_wrist_view: bool = True,
        use_depth:bool = False,
        dino_variant: str = "dinov2_vits14",
    ):
        super().__init__()
        self.finetune_type = finetune_type
        self.device = clip_device
        self.sequence_length = sequence_length
        self.action_pred_steps = action_pred_steps
        self.obs_pred = obs_pred
        self.atten_goal = atten_goal
        self.atten_goal_state = atten_goal_state
        self.atten_only_obs = atten_only_obs
        self.attn_robot_proprio_state = attn_robot_proprio_state
        self.mask_l_obs_ratio = mask_l_obs_ratio
        self.phase = phase
        assert self.phase in ["pretrain", "finetune", "evaluate"]
        self.gripper_width = gripper_width
        self.vit_checkpoint_path = vit_checkpoint_path

        # --- NEW: store switches ---
        self.encoder_type = encoder_type.lower()
        assert self.encoder_type in {"vit", "resnet"}
        self.use_text = use_text
        self.use_state = use_state
        self.use_wrist_view = use_wrist_view
        self.allow_obs_pred_with_resnet = allow_obs_pred_with_resnet
        self.use_depth = use_depth
        
        # --- apply preset if provided ---
        if model_size is not None:
            preset = SEER_PRESETS[model_size]
            hidden_dim = preset.hidden_dim
            transformer_layers = preset.transformer_layers
            transformer_heads = preset.transformer_heads
            num_resampler_query = preset.num_resampler_query
            num_obs_token_per_image = preset.num_obs_token_per_image
            perceiver_depth = preset.perceiver_depth
            self.dino_variant = preset.dino_variant
        else:
            perceiver_depth = 3  # default as before
            self.dino_variant = dino_variant

        self.hidden_dim = hidden_dim
        

        # text projector
        self.text_projector = nn.Linear(512, self.hidden_dim)

        # state encoders & projector
        ARM_STATE_FEATURE_DIM = self.hidden_dim
        GRIPPER_STATE_FEATURE_DIM = self.hidden_dim
        self.arm_state_encoder = nn.Linear(6, ARM_STATE_FEATURE_DIM)
        self.gripper_state_encoder = nn.Linear(2, GRIPPER_STATE_FEATURE_DIM)
        self.state_projector = nn.Linear(
            ARM_STATE_FEATURE_DIM + GRIPPER_STATE_FEATURE_DIM, self.hidden_dim
        )

        # action encoders (if needed elsewhere) & projector
        self.action_pose_encoder = nn.Linear(6, ARM_STATE_FEATURE_DIM)
        self.action_gripper_position_encoder = nn.Linear(2, GRIPPER_STATE_FEATURE_DIM)
        self.action_projector = nn.Linear(
            ARM_STATE_FEATURE_DIM + GRIPPER_STATE_FEATURE_DIM, self.hidden_dim
        )

        # ---------------- IMAGE ENCODER(S) ----------------
        self.NUM_OBS_TOKEN_PER_IMAGE = num_obs_token_per_image
        self.NUM_OBS_TOKEN = self.NUM_OBS_TOKEN_PER_IMAGE * (2 if self.use_wrist_view else 1)
        
        if self.use_depth:
            # --- Path 1: Depth-Aware CLIP-RoPE Embedder ---
            assert self.hidden_dim % 6 == 0, "For 3D RoPE, hidden_dim must be divisible by 6."
            
            # Store FPS counts to be used by _compute_num_A
            self.num_fps_samples_primary = 32
            self.num_fps_samples_wrist = 32
            
            self.image_primary_embedder_depth = RGBD_CLIP_RoPE_Embedder(
                feature_dim=self.hidden_dim, num_fps_samples=self.num_fps_samples_primary
            )
            if self.use_wrist_view:
                self.image_wrist_embedder_depth = RGBD_CLIP_RoPE_Embedder(
                    feature_dim=self.hidden_dim, num_fps_samples=self.num_fps_samples_wrist
                )
            print("[INFO] Using Depth-Aware CLIP-RoPE Embedder.")
            self.vision_encoder = None
            self.perceiver_resampler = None
            self.NUM_RESAMPLER_QUERY = 0

        elif self.encoder_type == "vit":
            # -------------------- DINOv2 VISION ENCODER --------------------
            # choose variant; default to ViT-S/14 (small)
            from dinov2.models.vision_transformer import vit_small, vit_base, vit_large

            # dino_variant can be passed as a string arg
            dino_variant = getattr(self, "dino_variant", "dinov2_vits14").lower()

            if dino_variant == "dinov2_vits14":
                self.vision_encoder = vit_small(patch_size=14)
                vit_embed_dim = 384
            elif dino_variant == "dinov2_vitb14":
                self.vision_encoder = vit_base(patch_size=14)
                vit_embed_dim = 768
            elif dino_variant == "dinov2_vitl14":
                self.vision_encoder = vit_large(patch_size=14)
                vit_embed_dim = 1024
            else:
                raise ValueError(f"Unsupported DINOv2 variant: {dino_variant}")


            for p in self.vision_encoder.parameters():
                p.requires_grad = False  # freeze encoder

            # -------------------- PERCEIVER RESAMPLER --------------------
            # The Perceiver now operates on DINO token outputs (CLS + patch tokens)
            self.RESAMPLER_hidden_dim = vit_embed_dim
            self.NUM_RESAMPLER_QUERY = num_resampler_query
            self.perceiver_resampler = PerceiverResampler(
                dim=self.RESAMPLER_hidden_dim,
                num_latents=self.NUM_RESAMPLER_QUERY,
                depth=perceiver_depth,
            )

            # Project DINO tokens to your hidden_dim space
            self.image_primary_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
            self.image_wrist_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
            self.cls_token_primary_projector = nn.Linear(vit_embed_dim, self.hidden_dim)
            self.cls_token_wrist_projector = nn.Linear(vit_embed_dim, self.hidden_dim)
            print(f"[INFO] Using DINOv2 encoder: {dino_variant} (embed dim = {vit_embed_dim})")

        else:  # RESNET path: 1 token per view, no Perceiver, no CLS tokens
            self.RESNET_OUT_DIM = 512
            # build the chosen resnet
            if resnet_variant == "resnet18":
                rn = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT if resnet_pretrained else None)
            elif resnet_variant == "resnet34":
                rn = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT if resnet_pretrained else None)
            elif resnet_variant == "resnet50":
                rn = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT if resnet_pretrained else None)
                self.RESNET_OUT_DIM = 2048
            else:
                raise ValueError(f"Unsupported resnet_variant {resnet_variant}")
            # keep everything up to the global pool (avgpool) + flatten
            rn.fc = nn.Identity()
            self.vision_encoder = rn
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

            # projectors for pooled features
            self.image_primary_projector = nn.Linear(self.RESNET_OUT_DIM, self.hidden_dim)
            if self.use_wrist_view:
                self.image_wrist_projector  = nn.Linear(self.RESNET_OUT_DIM, self.hidden_dim)

            # Perceiver-related placeholders to keep type checks safe
            self.perceiver_resampler = None
            self.NUM_RESAMPLER_QUERY = 0

        # --------------- PRED TOKENS / MASK ---------------
        if self.action_pred_steps > 0:
            self.action_pred_token = nn.Parameter(
                torch.zeros(1, 1, self.action_pred_steps, self.hidden_dim)
            )

        if self.obs_pred:
            self.obs_tokens = nn.Parameter(
                torch.zeros(1, 1, self.NUM_OBS_TOKEN if self.use_wrist_view else self.NUM_OBS_TOKEN_PER_IMAGE, self.hidden_dim)
            )

        # causal transformer
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_dim)

        # build initial attention mask using helper (will rebuild in forward, too)
        num_A = self._compute_num_A()
        this_num_obs_token = (self.NUM_OBS_TOKEN if self.obs_pred else 0)
        self.attention_mask = nn.Parameter(
            generate_attention_mask(
                K=self.sequence_length,
                num_A=num_A,
                num_B=this_num_obs_token + self.action_pred_steps,
                atten_goal=self.atten_goal,
                atten_goal_state=self.atten_goal_state,
                atten_only_obs=self.atten_only_obs,
                attn_robot_proprio_state=self.attn_robot_proprio_state,
                mask_l_obs_ratio=self.mask_l_obs_ratio,
                num_obs_token=this_num_obs_token,
                action_pred_steps=self.action_pred_steps,
            ),
            requires_grad=False,
        )

        # positional embedding per timestep
        self.transformer_backbone_position_embedding = nn.Parameter(
            torch.zeros(1, self.sequence_length, 1, self.hidden_dim), requires_grad=True
        )

        # GPT-2
        config = GPT2Config()
        config.hidden_size = self.hidden_dim
        config.n_layer = transformer_layers
        config.vocab_size = 1
        config.n_head = transformer_heads
        self.transformer_backbone = GPT2Model(config)

        # action decoder
        MLP_hidden_dim = self.hidden_dim // 2
        self.action_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, MLP_hidden_dim), nn.ReLU(),
            nn.Linear(MLP_hidden_dim, MLP_hidden_dim), nn.ReLU(),
        )
        self.arm_action_decoder = nn.Sequential(nn.Linear(MLP_hidden_dim, 6), torch.nn.Tanh())
        self.gripper_action_decoder = nn.Sequential(nn.Linear(MLP_hidden_dim, 1), torch.nn.Sigmoid())

        # (optional) state reconstruction heads (kept as-is)
        self.recon_state_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, MLP_hidden_dim), nn.ReLU(),
            nn.Linear(MLP_hidden_dim, MLP_hidden_dim), nn.ReLU(),
        )
        self.recon_arm_state_decoder = nn.Sequential(nn.Linear(MLP_hidden_dim, 6), torch.nn.Tanh())
        self.recon_gripper_state_decoder = nn.Sequential(nn.Linear(MLP_hidden_dim, 1), torch.nn.Sigmoid())

        # image decoder (obs_pred)
        self.IMAGE_DECODER_hidden_dim = self.hidden_dim
        self.NUM_MASK_TOKEN = int(calvin_input_image_size**2 / patch_size / patch_size)
        self.PATCH_SIZE = patch_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.IMAGE_DECODER_hidden_dim))
        self.image_decoder_obs_pred_projector = nn.Linear(self.hidden_dim, self.IMAGE_DECODER_hidden_dim)

        self.image_decoder_position_embedding = nn.Parameter(
            torch.zeros(1, self.NUM_OBS_TOKEN_PER_IMAGE + self.NUM_MASK_TOKEN, self.IMAGE_DECODER_hidden_dim),
            requires_grad=False
        )
        # auto-slim heads for small/tiny
        image_decoder_heads = 8 if self.hidden_dim <= 256 else 16
        self.image_decoder = nn.Sequential(
            Block(self.IMAGE_DECODER_hidden_dim, num_heads=image_decoder_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            Block(self.IMAGE_DECODER_hidden_dim, num_heads=image_decoder_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
        )
        self.image_decoder_norm = nn.LayerNorm(self.IMAGE_DECODER_hidden_dim)
        self.image_decoder_pred = nn.Linear(self.IMAGE_DECODER_hidden_dim, self.PATCH_SIZE**2 * 3)

        # initialize network
        self.initialize_weights()

        # # load CLIP text encoder (frozen)
        if os.path.exists("checkpoints/clip/ViT-B-32.pt"):
            self.clip_model, self.image_processor = clip.load("checkpoints/clip/ViT-B-32.pt", device=clip_device)
        else:
            self.clip_model, self.image_processor = clip.load("ViT-B/32", device=clip_device)
        for p in self.clip_model.parameters():
            p.requires_grad = False

    # ---- helpers ------------------------------------------------------------
    def _compute_num_A(self):
        """Compute num_A (non-learnable tokens per timestep) based on active modalities."""
        # text, state
        t = 1 if self.use_text else 0
        s = 1 if self.use_state else 0
        # image tokens
        if self.use_depth:
            # For the depth path, tokens are determined by FPS samples
            img_latents = self.num_fps_samples_primary
            if self.use_wrist_view:
                img_latents += self.num_fps_samples_wrist
            cls_tokens = 0 # No CLS tokens in this path

        elif self.encoder_type == "vit":
            views = 2 if self.use_wrist_view else 1
            img_latents = self.NUM_RESAMPLER_QUERY * views
            cls_tokens = views  # one CLS per view
        else:  # resnet: 1 pooled token per used view; no CLS
            img_latents = (2 if self.use_wrist_view else 1)
            cls_tokens = 0
        return t + s + img_latents + cls_tokens

    def initialize_weights(self):
        image_decoder_position_embedding_obs = get_2d_sincos_pos_embed(
            self.IMAGE_DECODER_hidden_dim, int(self.NUM_OBS_TOKEN_PER_IMAGE**.5), cls_token=False
        )
        image_decoder_position_embedding_mask = get_2d_sincos_pos_embed(
            self.IMAGE_DECODER_hidden_dim, int(self.NUM_MASK_TOKEN**.5), cls_token=False
        )
        image_decoder_position_embedding = np.concatenate(
            (image_decoder_position_embedding_obs, image_decoder_position_embedding_mask), axis=0
        )
        self.image_decoder_position_embedding.data.copy_(
            torch.from_numpy(image_decoder_position_embedding).float().unsqueeze(0)
        )
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.transformer_backbone_position_embedding, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_model_type(self):
        """Caches the torch dtype of key model components for later mixed-precision type casting."""
        # Vision encoder type (may differ for DINOv2 / ResNet)
        if self.use_depth:
            # For the depth path, get the dtype from the new embedder
            self.vision_encoder_type = next(self.image_primary_embedder_depth.parameters()).type()
        else:
            self.vision_encoder_type = next(self.vision_encoder.parameters()).type()

        # Some encoders (e.g. ResNet) don’t have a perceiver
        if hasattr(self, "perceiver_resampler") and self.perceiver_resampler is not None:
            self.perceiver_resampler_type = next(self.perceiver_resampler.parameters()).type()
        else:
            self.perceiver_resampler_type = self.vision_encoder_type

        # Transformer backbone (GPT2-style)
        self.transformer_backbone_type = next(self.transformer_backbone.parameters()).type()

        # Action decoder
        self.action_decoder_type = next(self.action_decoder.parameters()).type()

        # Optional: print for debugging
        print(f"[ModelType] vision={self.vision_encoder_type}, perceiver={self.perceiver_resampler_type}, "
            f"transformer={self.transformer_backbone_type}, action_decoder={self.action_decoder_type}")

    
    def _encode_vit_tokens(self, x: torch.Tensor):
        """
        Returns (cls_token: (B,1,D), patch_tokens: (B,N,D)) for ViT-like encoders.
        Supports DINOv2 (torchvision) and falls back gracefully.
        """
        with torch.no_grad():
            # Case 1: MAE-style (kept for safety/back-compat)
            if hasattr(self.vision_encoder, "forward_encoder"):
                toks, _, _ = self.vision_encoder.forward_encoder(x, mask_ratio=0.0)  # (B, 1+N, D)
                return toks[:, :1, :], toks[:, 1:, :]

            # Case 2: DINOv2 / ViT-style encoders in torchvision/timm
            if hasattr(self.vision_encoder, "forward_features"):
                out = self.vision_encoder.forward_features(x)
                # torchvision dinov2 returns a dict with normalized tokens
                if isinstance(out, dict):
                    if ("x_norm_clstoken" in out) and ("x_norm_patchtokens" in out):
                        cls = out["x_norm_clstoken"]              # (B, D)
                        patch = out["x_norm_patchtokens"]         # (B, N, D)
                        return cls.unsqueeze(1), patch
                    # some models return 'last_hidden_state'
                    if "last_hidden_state" in out:
                        hs = out["last_hidden_state"]             # (B, 1+N, D)
                        return hs[:, :1, :], hs[:, 1:, :]

                # some timm models return a single tensor (B,1+N,D)
                if torch.is_tensor(out) and out.dim() == 3 and out.size(1) >= 2:
                    return out[:, :1, :], out[:, 1:, :]

            # Case 3: Fallback — only a pooled feature is available
            pooled = self.vision_encoder(x)  # (B, D) usually
            return pooled.unsqueeze(1), None

    # ------------------------------------------------------------------------

    def forward(self, image_primary, image_wrist, state, text_token, action=None, 
            images_primary_depth=None, images_wrist_depth=None, 
            intrinsics_primary=None, intrinsics_wrist=None,
            extrinsics_primary=None, extrinsics_wrist=None):
        # rebuild attention mask each forward during training
        if self.training and self.phase == "pretrain":
            this_num_obs_token = (self.NUM_OBS_TOKEN if self.obs_pred else 0)
            self.attention_mask = nn.Parameter(
                generate_attention_mask(
                    K=self.sequence_length,
                    num_A=self._compute_num_A(),
                    num_B=this_num_obs_token + self.action_pred_steps,
                    atten_goal=self.atten_goal,
                    atten_goal_state=self.atten_goal_state,
                    atten_only_obs=self.atten_only_obs,
                    attn_robot_proprio_state=self.attn_robot_proprio_state,
                    mask_l_obs_ratio=self.mask_l_obs_ratio,
                    num_obs_token=this_num_obs_token,
                    action_pred_steps=self.action_pred_steps,
                ).to(self.device),
                requires_grad=False,
            )

        B, S, _ = state.shape
        device = image_primary.device
        S_AND_FUTURE = image_primary.shape[1]

        image_pred = None
        arm_pred_action, gripper_pred_action = None, None
        arm_pred_state, gripper_pred_state = None, None
        loss_arm_action = None

        # -------- text embedding --------
        text_embedding = None
        if self.use_text:
            with torch.no_grad():
                text_feature = self.clip_model.encode_text(text_token.flatten(0, 1))
                text_feature = text_feature.type(state.type())
            text_embedding = self.text_projector(text_feature).view(B, S, -1, self.hidden_dim)

        # -------- state embedding --------
        state_embedding = None
        if self.use_state:
            state_flat = state.flatten(0, 1)
            arm_state_feature = self.arm_state_encoder(state_flat[:, :6])
            if not self.gripper_width:
                gripper_state_one_hot = torch.nn.functional.one_hot(
                    torch.where(state_flat[:, 6:].flatten() < 1,
                                torch.tensor(0, device=device),
                                torch.tensor(1, device=device)),
                    num_classes=2
                )
                gripper_state_feature = self.gripper_state_encoder(gripper_state_one_hot.type_as(state_flat))
            else:
                gripper_state_feature = self.gripper_state_encoder(state_flat[:, 6:])
            state_embedding = self.state_projector(torch.cat((arm_state_feature, gripper_state_feature), dim=1))
            state_embedding = state_embedding.view(B, S, -1, self.hidden_dim)

        # -------- image features --------
        if self.use_depth:
            # --- Path 1: Depth-Aware CLIP-RoPE Embedder ---
            assert images_primary_depth is not None and intrinsics_primary is not None and extrinsics_primary is not None, \
                "Depth images and intrinsics must be provided when use_depth is True."
            # Match dtype with encoder
            enc_param = next(self.image_primary_embedder_depth.parameters())
            if image_primary.type() != enc_param.type():
                image_primary = image_primary.type(enc_param.type())
                if self.use_wrist_view:
                    image_wrist = image_wrist.type(enc_param.type())

            ip_flat = image_primary.flatten(0, 1)
            ipd_flat = images_primary_depth.flatten(0, 1)

            # print("debug: Seer agentmini:forward:ip_flat", ip_flat.shape, ip_flat.dtype)
            # print("debug: Seer agentmini:forward:ipd_flat", ipd_flat.shape, ipd_flat.dtype)
            
            # Intrinsics are static, so we repeat them for the sequence
            intrinsics_primary_seq = intrinsics_primary.flatten(0, 1) #.unsqueeze(1).repeat(1, S_AND_FUTURE, 1, 1).flatten(0, 1)
            # print("debug: Seer agentmini:forward:intrinsic", intrinsics_primary_seq.shape, intrinsics_primary_seq.dtype)
            
            # Extrinsics are also static for the primary cam, so we repeat
            extrinsics_primary_seq = extrinsics_primary.flatten(0, 1) #.unsqueeze(1).repeat(1, S_AND_FUTURE, 1, 1).flatten(0, 1)
            # print("debug: Seer agentmini:forward:extrinsic", extrinsics_primary_seq.shape, extrinsics_primary_seq.dtype)

            ip_emb_flat = self.image_primary_embedder_depth(
                ip_flat, ipd_flat, intrinsics_primary_seq, extrinsics_primary_seq
            )
            num_ip_tokens = ip_emb_flat.shape[1]
            ip_emb = ip_emb_flat.view(B, S_AND_FUTURE, num_ip_tokens, self.hidden_dim)[:, :S, :, :]

            # --- WRIST CAMERA ---
            if self.use_wrist_view:
                # extrinsics_wrist has shape (B, S_AND_FUTURE, 4, 4)
                iw_flat = image_wrist.flatten(0, 1)
                iwd_flat = images_wrist_depth.flatten(0, 1)
                
                # Intrinsics are static for the wrist cam sensor itself
                intrinsics_wrist_seq = intrinsics_wrist.flatten(0, 1) #.unsqueeze(1).repeat(1, S_AND_FUTURE, 1, 1).flatten(0, 1)

                extrinsics_wrist_seq = extrinsics_wrist.flatten(0, 1)

                iw_emb_flat = self.image_wrist_embedder_depth(
                    iw_flat, iwd_flat, intrinsics_wrist_seq, extrinsics_wrist_seq
                )
                
                num_iw_tokens = iw_emb_flat.shape[1]
                iw_emb = iw_emb_flat.view(B, S_AND_FUTURE, num_iw_tokens, self.hidden_dim)[:, :S, :, :]
                image_embedding = torch.cat((ip_emb, iw_emb), dim=2)
            else:
                image_embedding = ip_emb
                
            image_cls_token_embedding = None

        elif self.encoder_type == "vit":
            # Match dtype with encoder
            enc_param = next(self.vision_encoder.parameters())
            if image_primary.type() != enc_param.type():
                image_primary = image_primary.type(enc_param.type())
                if self.use_wrist_view:
                    image_wrist = image_wrist.type(enc_param.type())

            # DINOv2 (or MAE back-compat) -> CLS + patch tokens
            ip_cls, ip_patch = self._encode_vit_tokens(image_primary.flatten(0, 1))  # (B*S+F,1,D), (B*S+F,N,D)
            if self.use_wrist_view:
                iw_cls, iw_patch = self._encode_vit_tokens(image_wrist.flatten(0, 1))
            else:
                iw_cls, iw_patch = None, None

            # reshape back to (B, S+F, 1/N, D)
            ip_cls = ip_cls.view(B, S_AND_FUTURE, 1, -1)
            if ip_patch is not None:
                num_patches = ip_patch.shape[1]              # e.g., 256 for 224px, patch14
                ip_patch = ip_patch.view(B, S_AND_FUTURE, num_patches, -1)
            else:
                num_patches = 0

            if self.use_wrist_view:
                iw_cls = iw_cls.view(B, S_AND_FUTURE, 1, -1)
                if iw_patch is not None:
                    iw_patch = iw_patch.view(B, S_AND_FUTURE, iw_patch.shape[1], -1)

            # Perceiver over *patch* tokens only (past S steps, not future)
            if ip_patch is not None:
                ip_lat = self.perceiver_resampler(
                    ip_patch[:, :S, :, :].reshape(B * S, num_patches, self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1)
                )
                ip_emb = self.image_primary_projector(ip_lat.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
            else:
                # If only CLS is available, fall back to a single image token
                ip_emb = self.image_primary_projector(ip_cls[:, :S, :, :].flatten(0, 2)).view(B, S, 1, self.hidden_dim)

            if self.use_wrist_view:
                if iw_patch is not None:
                    iw_lat = self.perceiver_resampler(
                        iw_patch[:, :S, :, :].reshape(B * S, iw_patch.shape[2], self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1)
                    )
                    iw_emb = self.image_wrist_projector(iw_lat.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
                else:
                    iw_emb = self.image_wrist_projector(iw_cls[:, :S, :, :].flatten(0, 2)).view(B, S, 1, self.hidden_dim)
                image_embedding = torch.cat((ip_emb, iw_emb), dim=2)
            else:
                image_embedding = ip_emb

            # CLS token embeddings
            ip_cls_emb = self.cls_token_primary_projector(ip_cls[:, :S, :, :].flatten(0, 2)).view(B, S, -1, self.hidden_dim)
            if self.use_wrist_view:
                iw_cls_emb = self.cls_token_wrist_projector(iw_cls[:, :S, :, :].flatten(0, 2)).view(B, S, -1, self.hidden_dim)
                image_cls_token_embedding = torch.cat((ip_cls_emb, iw_cls_emb), dim=2)
            else:
                image_cls_token_embedding = ip_cls_emb

        else:
            # RESNET pooled features -> 1 token per view
            rn_type = next(self.vision_encoder.parameters()).type()
            if image_primary.type() != rn_type:
                image_primary = image_primary.type(rn_type)
                if self.use_wrist_view:
                    image_wrist = image_wrist.type(rn_type)

            with torch.no_grad():
                ip_feat = self.vision_encoder(image_primary.flatten(0, 1))  # (B*S+F, C)
                if self.use_wrist_view:
                    iw_feat = self.vision_encoder(image_wrist.flatten(0, 1))

            ip_emb = self.image_primary_projector(ip_feat).view(B, S_AND_FUTURE, 1, self.hidden_dim)[:, :S, :, :]
            if self.use_wrist_view:
                iw_emb = self.image_wrist_projector(iw_feat).view(B, S_AND_FUTURE, 1, self.hidden_dim)[:, :S, :, :]
                image_embedding = torch.cat((ip_emb, iw_emb), dim=2)  # (B,S, views, D)
            else:
                image_embedding = ip_emb

            image_cls_token_embedding = None  # no CLS in ResNet path

            # if obs_pred not allowed with resnet, disable it
            if self.obs_pred and not self.allow_obs_pred_with_resnet:
                self.obs_pred = False

        # -------- aggregate embeddings --------
        embed_list = []
        if self.use_text and text_embedding is not None:
            embed_list.append(text_embedding)
        if self.use_state and state_embedding is not None:
            embed_list.append(state_embedding)
        embed_list.append(image_embedding)
        if image_cls_token_embedding is not None:
            embed_list.append(image_cls_token_embedding)

        embeddings = torch.cat(embed_list, dim=2)  # (B,S,T,D)
        pred_token_start_idx = embeddings.shape[2]

        trans_in_list = [embeddings]
        if self.obs_pred:
            # ensure correct NUM_OBS_TOKEN under wrist toggle
            this_num_obs_token = self.NUM_OBS_TOKEN if self.use_wrist_view else self.NUM_OBS_TOKEN_PER_IMAGE
            trans_in_list.append(self.obs_tokens[:, :, :this_num_obs_token, :].repeat(B, S, 1, 1))
        if self.action_pred_steps > 0:
            trans_in_list.append(self.action_pred_token.repeat(B, S, 1, 1))
        transformer_input = torch.cat(trans_in_list, dim=2)
        transformer_input = transformer_input + self.transformer_backbone_position_embedding.repeat(
            B, 1, transformer_input.shape[-2], 1
        )
        transformer_input = transformer_input.flatten(1, 2)

        # transformer
        transformer_input = self.embedding_layer_norm(transformer_input)
        transformer_output = self.transformer_backbone(
            inputs_embeds=transformer_input, attention_mask=self.attention_mask
        )
        transformer_output = transformer_output.view(B, S, -1, self.hidden_dim)

        # -------- obs_pred (image tokens -> decoder) --------
        if self.obs_pred:
            this_num_obs_token = self.NUM_OBS_TOKEN if self.use_wrist_view else self.NUM_OBS_TOKEN_PER_IMAGE
            obs_pred_feature = transformer_output[:, :, pred_token_start_idx : pred_token_start_idx + this_num_obs_token, :]
            obs_pred_embedding = self.image_decoder_obs_pred_projector(obs_pred_feature.reshape(-1, self.hidden_dim))
            groups = this_num_obs_token // self.NUM_OBS_TOKEN_PER_IMAGE
            obs_pred_embedding = obs_pred_embedding.view(B * S * groups, self.NUM_OBS_TOKEN_PER_IMAGE, self.IMAGE_DECODER_hidden_dim)
            mask_tokens = self.mask_token.repeat(B * S * groups, self.NUM_MASK_TOKEN, 1)
            image_decoder_input = torch.cat((obs_pred_embedding, mask_tokens), dim=1)
            image_decoder_input = image_decoder_input + self.image_decoder_position_embedding
            image_decoder_output = self.image_decoder(image_decoder_input)
            image_pred_feature = image_decoder_output[:, -self.NUM_MASK_TOKEN:, :]
            image_pred_feature = self.image_decoder_norm(image_pred_feature.reshape(-1, self.IMAGE_DECODER_hidden_dim))
            image_pred = self.image_decoder_pred(image_pred_feature)
            image_pred = image_pred.view(B * S, groups, self.NUM_MASK_TOKEN, -1)

        # -------- action prediction --------
        idx_start = pred_token_start_idx + (this_num_obs_token if self.obs_pred else 0)
        if self.action_pred_steps > 0:
            action_pred_feature = transformer_output[:, :, idx_start: idx_start + self.action_pred_steps, :]
            action_pred_feature = self.action_decoder(action_pred_feature)
            arm_pred_action = self.arm_action_decoder(action_pred_feature)
            gripper_pred_action = self.gripper_action_decoder(action_pred_feature)

        return arm_pred_action, gripper_pred_action, image_pred, arm_pred_state, gripper_pred_state, loss_arm_action
