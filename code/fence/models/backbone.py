"""
FENCE Backbone with Spectral Normalization and Projection Head.
"""

import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToPILImage

try:
    import timm
except ImportError:
    timm = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from fence.config import (
        FENCE_BACKBONE, FENCE_EMBEDDING_DIM,
        SPECTRAL_NORM_ENABLED, SPECTRAL_NORM_ITERATIONS,
        DEVICE, INPUT_NORMALIZED
    )
except ImportError:
    print("Warning: fence.config not found. Using defaults.")
    FENCE_BACKBONE = "vit_large_patch14_224"
    FENCE_EMBEDDING_DIM = 768
    SPECTRAL_NORM_ENABLED = True
    SPECTRAL_NORM_ITERATIONS = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_NORMALIZED = False


class FENCEBackbone(nn.Module):
    def __init__(self,
                backbone_name: str = FENCE_BACKBONE,
                embedding_dim: int = FENCE_EMBEDDING_DIM,
                pretrained: bool = True,
                spectral_norm: bool = SPECTRAL_NORM_ENABLED,
                spectral_norm_iterations: int = SPECTRAL_NORM_ITERATIONS,
                use_lora: bool = False,
                lora_rank: int = 32,
                lora_alpha: int = 64,
                use_ssf: bool = True,
                device: str = DEVICE):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.spectral_norm = spectral_norm
        self.spectral_norm_iterations = spectral_norm_iterations
        self.device = device
        self.to_pil = ToPILImage()
        self.input_normalized = INPUT_NORMALIZED

        # Normalization buffers. The dataloader hands us ImageNet-normalized
        # tensors; CLIP expects its own mean/std. We remap on the fly inside
        # forward() so gradients can flow through the CLIP tower directly.
        self.register_buffer('_imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer('_clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('_clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        self.backbone_type = self._get_backbone_type()  # <-- THIS METHOD MUST EXIST
        print(f"Initializing FENCEBackbone:")
        print(f"  Backbone: {backbone_name}")
        print(f"  Type: {self.backbone_type}")
        print(f"  Embedding Dim: {embedding_dim}")
        print(f"  Spectral Norm: {spectral_norm}")
        
        # Initialize backbone
        self.backbone = self._init_backbone(pretrained)
        
        # --- PROJECTION HEAD ---
        feature_dim = self._get_feature_dim()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )
        print(f"  Projection head: {feature_dim} -> 512 -> {embedding_dim}")
        
        # Apply spectral normalization (silent)
        if self.spectral_norm:
            self._apply_spectral_norm(silent=True)
        
        self.to(device)
        self._print_trainable_params()

    # ========================================================================
    # 1. BACKBONE TYPE DETECTION (THIS WAS MISSING)
    # ========================================================================
    
    def _get_backbone_type(self) -> str:
        """Determine the backbone type from the name."""
        name = self.backbone_name.lower()
        if 'vit' in name:
            return 'vit'
        elif 'convnext' in name:
            return 'convnext'
        elif 'resnet' in name or 'efficientnet' in name or 'mobilenet' in name:
            return 'timm'
        else:
            return 'timm'  # Default to timm

    # ========================================================================
    # 2. BACKBONE INITIALIZATION
    # ========================================================================
    
    def _init_backbone(self, pretrained: bool):
        """Initialize the backbone model."""
        if self.backbone_type == 'vit':
            return self._init_vit_backbone(pretrained)
        elif self.backbone_type == 'convnext':
            return self._init_convnext_backbone(pretrained)
        else:
            return self._init_timm_backbone(pretrained)

    # ========================================================================
    # 3. VIT BACKBONE
    # ========================================================================
    
    def _init_vit_backbone(self, pretrained):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers required")
        model_map = {
            'vit_large_patch14_224': 'clip-ViT-L-14',
            'vit_base_patch16_224': 'clip-ViT-B-16',
            'vit_base_patch32_224': 'clip-ViT-B-32',
        }
        st_name = model_map.get(self.backbone_name, 'clip-ViT-L-14')
        print(f"  Loading {st_name}")
        self.clip_model = SentenceTransformer(st_name, device=self.device)

        # Freeze everything
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze last 6 transformer blocks + layernorm + visual projection
        vision_model = None
        try:
            vision_model = self.clip_model[0].model.vision_model
        except AttributeError:
            try:
                vision_model = self.clip_model[0].model.visual
            except AttributeError:
                raise RuntimeError(
                    "Could not locate vision_model in SentenceTransformer CLIP. "
                    "Please check the module structure."
                )

        if vision_model is None:
            raise RuntimeError("vision_model not found. Aborting to avoid silent failure.")

        if hasattr(vision_model, 'encoder') and hasattr(vision_model.encoder, 'layers'):
            layers = vision_model.encoder.layers
            num_layers = len(layers)
            for i in range(max(0, num_layers - 6), num_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
            if hasattr(vision_model, 'layernorm'):
                for param in vision_model.layernorm.parameters():
                    param.requires_grad = True
            if hasattr(self.clip_model[0].model, 'visual_projection'):
                for param in self.clip_model[0].model.visual_projection.parameters():
                    param.requires_grad = True
            print(f"  Unfroze last 6 transformer blocks + layernorm + projection")
        else:
            raise RuntimeError("vision_model.encoder.layers not found.")

        # Gradient checkpointing: trade compute for memory so CLIP-L fine-tuning
        # (with two contrastive views) fits on a single GPU. use_reentrant=False
        # is required because the inputs to the unfrozen blocks do not themselves
        # require grad (the earlier blocks are frozen).
        try:
            self.clip_model[0].model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print(f"  Gradient checkpointing enabled (use_reentrant=False)")
        except TypeError:
            try:
                self.clip_model[0].model.gradient_checkpointing_enable()
                print(f"  Gradient checkpointing enabled")
            except Exception as e:
                print(f"  Note: gradient checkpointing not enabled: {e}")
        except Exception as e:
            print(f"  Note: gradient checkpointing not enabled: {e}")

        self._vision_model = vision_model
        self._visual_projection = self.clip_model[0].model.visual_projection
        return self.clip_model

    # ========================================================================
    # 4. CONVNEXT BACKBONE
    # ========================================================================
    
    def _init_convnext_backbone(self, pretrained: bool):
        if timm is None:
            raise ImportError("timm required for ConvNeXt")
        
        model_map = {
            'convnext_v2_large': 'convnextv2_large',
            'convnext_v2_base': 'convnextv2_base',
            'convnext_v2_huge': 'convnextv2_huge',
        }
        timm_name = model_map.get(self.backbone_name, 'convnextv2_large')
        
        self.backbone = timm.create_model(
            timm_name, 
            pretrained=pretrained,
            num_classes=0, 
            global_pool='avg'
        )
        
        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the last block of the last stage
        try:
            stages = self.backbone.stages
            last_stage = stages[-1]
            last_block = last_stage.blocks[-1]
            for param in last_block.parameters():
                param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            print(f"  Unfroze last block of final stage")
        except Exception as e:
            print(f"  Warning: Could not unfreeze specific layers: {e}")
            layers = list(self.backbone.parameters())
            for param in layers[-10:]:
                param.requires_grad = True
            print(f"  Fallback: Unfroze last 10 layers")
        
        try:
            self.backbone.set_grad_checkpointing(True)
            print(f"  Gradient checkpointing enabled")
        except Exception as e:
            print(f"  Note: Gradient checkpointing not available: {e}")
        
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        return self.backbone

    # ========================================================================
    # 5. TIMM BACKBONE (ResNet, EfficientNet, MobileNet)
    # ========================================================================
    
    def _init_timm_backbone(self, pretrained: bool):
        if timm is None:
            raise ImportError("timm required for this backbone")
        
        print(f"  Loading timm model: {self.backbone_name}")
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        try:
            if hasattr(self.backbone, 'layer4'):
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
                print(f"  Unfroze layer4")
            elif hasattr(self.backbone, 'stages'):
                last_stage = self.backbone.stages[-1]
                for param in last_stage.parameters():
                    param.requires_grad = True
                print(f"  Unfroze last stage")
            elif hasattr(self.backbone, 'blocks'):
                for param in self.backbone.blocks[-1].parameters():
                    param.requires_grad = True
                print(f"  Unfroze last block")
            if hasattr(self.backbone, 'norm'):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
        except Exception as e:
            print(f"  Warning: Could not unfreeze specific layers: {e}")
            params = list(self.backbone.parameters())
            for param in params[-10:]:
                param.requires_grad = True
            print(f"  Fallback: Unfroze last 10 layers")
        
        return self.backbone

    # ========================================================================
    # 6. SPECTRAL NORM
    # ========================================================================
    
    def _apply_spectral_norm(self, silent: bool = False):
        """Apply spectral normalization to trainable linear/conv layers only."""
        def apply_sn(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.Linear, nn.Conv2d)):
                    if any(p.requires_grad for p in child.parameters()):
                        try:
                            setattr(module, name, utils.spectral_norm(child, 
                                n_power_iterations=self.spectral_norm_iterations))
                            if not silent:
                                print(f"  Spectral norm applied to: {name}")
                        except Exception:
                            pass
                else:
                    apply_sn(child)
        
        if self.backbone_type == 'vit':
            if hasattr(self, '_vision_model'):
                apply_sn(self._vision_model)
        else:
            apply_sn(self.backbone)

    # ========================================================================
    # 7. FEATURE DIM
    # ========================================================================
    
    def _get_feature_dim(self) -> int:
        """Get the feature dimension from the backbone."""
        if self.backbone_type == 'vit':
            return 768
        elif self.backbone_type == 'convnext':
            if 'large' in self.backbone_name.lower():
                return 1536
            elif 'base' in self.backbone_name.lower():
                return 1024
            else:
                return 768
        else:
            try:
                dummy = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    out = self.backbone(dummy)
                return out.shape[-1]
            except:
                return 2048

    # ========================================================================
    # 8. PARAM COUNTS
    # ========================================================================
    
    def _print_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # ========================================================================
    # 9. FORWARD PASS
    # ========================================================================
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == 'vit':
            # x is a normalized image tensor (B, 3, 224, 224).
            #
            # IMPORTANT: do NOT round-trip through PIL / clip_model.encode().
            #   - encode() runs under torch.no_grad() -> backbone never trains
            #   - to_pil() on a normalized tensor wraps pixels (mod 256) -> garbage
            #
            # Instead, re-normalize to CLIP's stats and call the HF CLIP image
            # tower directly so gradients flow into the unfrozen blocks.
            if self.input_normalized:
                pixel_values = x  # dataloader already produced CLIP-normalized input
            else:
                # incoming tensor is ImageNet-normalized -> undo, then CLIP-normalize
                x01 = x * self._imagenet_std + self._imagenet_mean
                pixel_values = (x01 - self._clip_mean) / self._clip_std

            # match the CLIP model's parameter dtype (fp32/fp16)
            clip_dtype = next(self.clip_model.parameters()).dtype
            pixel_values = pixel_values.to(dtype=clip_dtype)

            # get_image_features = vision_model -> pooled -> visual_projection
            # (all differentiable; honors the unfrozen blocks + spectral norm)
            features = self.clip_model[0].model.get_image_features(
                pixel_values=pixel_values
            )
        else:
            features = self.backbone(x)

        features = self.projection(features)
        features = F.normalize(features, p=2, dim=1)

        return features

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_lipschitz_constant(self):
        if not self.spectral_norm:
            return 1.0

        L = 1.0

        def top_sigma(weight: torch.Tensor) -> float:
            w = weight.detach()
            if w.dim() < 2:
                return 1.0
            w2d = w.view(w.shape[0], -1)
            try:
                s = torch.linalg.svdvals(w2d)
                return s[0].item()
            except Exception:
                return w2d.norm(p=2).item()

        def collect_sn(module):
            nonlocal L
            for name, child in module.named_children():
                if hasattr(child, 'weight_orig'):
                    L *= top_sigma(child.weight_orig)
                elif isinstance(child, (nn.Linear, nn.Conv2d)) and hasattr(child, 'weight'):
                    L *= top_sigma(child.weight)
                else:
                    collect_sn(child)

        if self.backbone_type == 'vit':
            collect_sn(self._vision_model)
        else:
            collect_sn(self.backbone)

        return L