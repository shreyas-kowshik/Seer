from functools import partial
import numpy as np
import torch
import torch.nn as nn
from clip.model import ModifiedResNet
import torch.nn.functional as F
import math
import clip
from timm.models.vision_transformer import PatchEmbed, Block
from utils.depth_utils import *

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

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # print("path_embed.device: ", self.patch_embed.device)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)  # qk_scale=None, 
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)  # qk_scale=None, 
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        # set_trace()
        # print("patch_embed cuda: ", next(self.patch_embed.parameters()).is_cuda)
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    

class ModifiedResNetFeatures(ModifiedResNet):
    def forward(self, x: torch.Tensor):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x0 = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # We only need the final feature map for this approach
        return x4

def load_clip_features():
    """Loads the modified CLIP RN50 as a feature extractor."""
    clip_model, clip_transforms = clip.load("RN50")
    state_dict = clip_model.state_dict()
    layers = tuple([len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]])
    output_dim = state_dict["text_projection"].shape[1]
    heads = state_dict["visual.layer1.0.conv1.weight"].shape[0] * 32 // 64
    backbone = ModifiedResNetFeatures(layers, output_dim, heads)
    backbone.load_state_dict(clip_model.visual.state_dict())
    normalize = clip_transforms.transforms[-1]
    # Freeze the backbone
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone, normalize

class RotaryPositionalEmbedding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0 and (dim // 3) % 2 == 0, \
            "feature_dim must be divisible by 2 and (dim//3) must be even (i.e., feature_dim % 6 == 0)."

        self.dim = dim
        self.x_dim = dim // 3
        self.y_dim = dim // 3
        self.z_dim = dim - 2 * (dim // 3)

        # (d/2,) each
        inv_x = 1.0 / (10000 ** (torch.arange(0, self.x_dim, 2).float() / self.x_dim))
        inv_y = 1.0 / (10000 ** (torch.arange(0, self.y_dim, 2).float() / self.y_dim))
        inv_z = 1.0 / (10000 ** (torch.arange(0, self.z_dim, 2).float() / self.z_dim))

        self.register_buffer("inv_freq_x", inv_x, persistent=False)
        self.register_buffer("inv_freq_y", inv_y, persistent=False)
        self.register_buffer("inv_freq_z", inv_z, persistent=False)

    def forward(self, xyz_coords, features):
        # xyz_coords: (B, S, 3), features: (B, S, D=self.dim)
        B, S, _ = xyz_coords.shape
        D = features.shape[-1]
        assert D == self.dim, f"features dim {D} != rope dim {self.dim}"
        # split features
        fx, fy, fz = torch.split(features, [self.x_dim, self.y_dim, self.z_dim], dim=-1)
        # ensure buffers are on the same device/dtype
        inv_x = self.inv_freq_x.to(features.device, dtype=features.dtype)
        inv_y = self.inv_freq_y.to(features.device, dtype=features.dtype)
        inv_z = self.inv_freq_z.to(features.device, dtype=features.dtype)

        # x,y,z: (B, S)
        x = xyz_coords[..., 0]
        y = xyz_coords[..., 1]
        z = xyz_coords[..., 2]

        # freq tensors: (B, S, d/2) using simple broadcasting (no einsum)
        freqs_x = x.unsqueeze(-1) * inv_x.view(1, 1, -1)
        freqs_y = y.unsqueeze(-1) * inv_y.view(1, 1, -1)
        freqs_z = z.unsqueeze(-1) * inv_z.view(1, 1, -1)
        # print("xyz cords shape: ",xyz_coords.shape)
        # print("feature shape: ", features.shape)
        # print("freqs x : ", freqs_x.shape)
        # print("freqs y : ", freqs_y.shape)
        # print("freqs z : ", freqs_z.shape)
        # print("fx : ", fx.shape)
        # print("fy  : ", fy.shape)
        # print("fz  : ", fz.shape)
        # apply rotary per axis
        fx = self._apply_rotary_emb(fx, freqs_x)
        fy = self._apply_rotary_emb(fy, freqs_y)
        fz = self._apply_rotary_emb(fz, freqs_z)

        return torch.cat((fx, fy, fz), dim=-1)

    @staticmethod
    def _apply_rotary_emb(features_axis, freqs):
        """
        features_axis: (B, S, d_axis), d_axis even
        freqs:         (B, S, d_axis/2)
        """
        # pair last dim into 2
        B, S, d = features_axis.shape
        assert d % 2 == 0, "axis dim must be even for rotary pairing"
        f = features_axis.reshape(B, S, d // 2, 2)        # (B, S, d/2, 2)

        c1 = freqs.cos().unsqueeze(-1).squeeze(-1)                      # (B, S, d/2, 1)
        s1 = freqs.sin().unsqueeze(-1).squeeze(-1) 
        # print("f shape: ",f.shape)
        # print("c shape: ", c1.shape)
        # print("s shape: ", s1.shape)
        try:
            x = f[..., 0] * c1 - f[..., 1] * s1
            y = f[..., 0] * s1 + f[..., 1] * c1
        except Exception as e:
            print("  [RoPE] EXCEPTION during rotary multiply:", repr(e))
            # print("  f:", f.shape, f.dtype, f.device)
            # print("  c:", c1.shape, c1.dtype, c1.device)
            # print("  s:", s1.shape, s1.dtype, s1.device)
            # # Drop to pdb so you can poke tensors
            # _pdb_here()
            raise

        return torch.stack((x.squeeze(-1), y.squeeze(-1)), dim=-1).flatten(-2)  # (B, S, d)

    

class RGBD_CLIP_RoPE_Embedder(nn.Module):
    def __init__(self, feature_dim, num_fps_samples=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_fps_samples = num_fps_samples

        # 1. Load pre-trained CLIP RN50 feature extractor
        self.clip_backbone, self.clip_normalize = load_clip_features()
        
        # 2. Projector to map CLIP features to the desired hidden dimension
        # The final layer of RN50 outputs 2048 channels
        self.feature_projector = nn.Linear(2048, self.feature_dim)
        
        # 3. RoPE Module
        self.rope = RotaryPositionalEmbedding3D(dim=self.feature_dim)
        print("RGBD Encoder: Final feature dim: ", feature_dim)

    def forward(self, rgb, depth, intrinsics, extrinsics):
        # Input shapes: rgb (B, C, H, W), depth (B, 1, H, W), intrinsics (B, 3, 3)
        B, _, H_in, W_in = rgb.shape
        
        # 1. Extract visual features from RGB
        # rgb_normalized = self.clip_normalize(rgb)
        visual_features = self.clip_backbone(rgb) # (B, 2048, H_feat, W_feat)
        
        # 2. Project to target dimension
        # (B, 2048, H_feat, W_feat) -> (B, H_feat*W_feat, 2048) -> (B, H_feat*W_feat, D)
        visual_tokens = visual_features.flatten(2).permute(0, 2, 1)
        projected_tokens = self.feature_projector(visual_tokens)

        # 3. Unproject depth to point cloud
        _, _, H_feat, W_feat = visual_features.shape
        depth_downsampled = F.interpolate(depth, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
        

        # Scale intrinsics to match downsampled resolution
        scale_factor = H_in / H_feat
        intrinsics_scaled = intrinsics.clone()
        intrinsics_scaled[:, :2, :] /= scale_factor
        
        point_cloud_camera_frame = depth_to_pointcloud(depth_downsampled, intrinsics_scaled)
        xyz_coords_camera_frame = point_cloud_camera_frame.flatten(1, 2) # (B, N, 3)
        
        Bf, Hf2, Wf2, _ = point_cloud_camera_frame.shape
        assert Hf2 == H_feat and Wf2 == W_feat
        assert xyz_coords_camera_frame.shape == (Bf, Hf2*Wf2, 3)

        if extrinsics is not None:
            # extrinsics is a (B, 4, 4) matrix [R | T]
            #                                  [0 | 1]
            R = extrinsics[:, :3, :3]  # (B, 3, 3) Rotation
            T = extrinsics[:, :3, 3]   # (B, 3) Translation

            # Apply transformation: P_world = R @ P_camera + T
            # (B, N, 3) = (B, 3, 3) @ (B, N, 3).transpose(1,2) -> (B, 3, N) -> transpose -> (B, N, 3)
            xyz_coords_world_frame = torch.bmm(xyz_coords_camera_frame, R.transpose(1, 2)) + T.unsqueeze(1)
            
            # This is now our definitive set of coordinates
            xyz_coords = xyz_coords_world_frame
        else:
            # If no extrinsics are provided, operate in the camera frame
            xyz_coords = xyz_coords_camera_frame

        # 4. Farthest Point Sampling (FPS) to select a subset of tokens
        # We sample based on 3D position for geometric coverage
        fps_indices = farthest_point_sample(xyz_coords, self.num_fps_samples) # (B, num_samples)
        assert fps_indices.dtype == torch.long
        # print("fps indices shape: ",fps_indices.shape)

        idx_tok = fps_indices.unsqueeze(-1).expand(-1, -1, projected_tokens.size(-1))  # (B, S, D)
        sampled_tokens = projected_tokens.gather(1, idx_tok)  # (B, S, D)
        # print("sampled tokens shape:" , sampled_tokens.shape)
        idx_xyz = fps_indices.unsqueeze(-1).expand(-1, -1, 3)  # (B, S, 3)
        sampled_xyz = xyz_coords.gather(1, idx_xyz)  # (B, S, 3)
        # print("sampled xyz shape: ", sampled_xyz.shape)
        # 5. Apply RoPE to the sampled tokens
        assert sampled_tokens.shape[:2] == sampled_xyz.shape[:2], \
        f"tokens {sampled_tokens.shape} vs xyz {sampled_xyz.shape}"

        embedded_tokens = self.rope(sampled_xyz, sampled_tokens)
        
        return embedded_tokens 

# import os, sys, torch

# def _is_rank0():
#     try:
#         import torch.distributed as dist
#         return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
#     except Exception:
#         return True

# def _pdb_here():
#     # use pdb++ if installed; else builtin pdb
#     if _is_rank0():
#         try:
#             import pdbpp as pdb  # type: ignore
#         except Exception:
#             import pdb
#         pdb.set_trace()
