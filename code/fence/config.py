"""
FENCE Configuration (Feasible ENclosure with Certified radius)

Inherits base paths, device, and GPU settings from root config.
Defines all FENCE-specific hyperparameters.

Usage:
    python fence/config.py  # Prints configuration and runs basic validation
"""

import sys
import importlib.util
from pathlib import Path

# Get the root directory (parent of code/)
ROOT_DIR = Path(__file__).parent.parent.parent
CODE_DIR = ROOT_DIR / "code"

# Manually load root config to avoid circular import
spec = importlib.util.spec_from_file_location("root_config", CODE_DIR / "config.py")
root_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_config)

# ============================================================================
# Inherit from root config
# ============================================================================
BASE_DIR = root_config.BASE_DIR
DEVICE = root_config.DEVICE
GPU_IDS = root_config.GPU_IDS
SEED = root_config.SEED
NUM_WORKERS = root_config.NUM_WORKERS
PIN_MEMORY = root_config.PIN_MEMORY
DATASET_NAME = root_config.DATASET_NAME
IMAGE_DIR = root_config.IMAGE_DIR
ANNOTATION_DIR = root_config.ANNOTATION_DIR
BATCH_SIZE = root_config.BATCH_SIZE
IMAGE_SIZE = root_config.IMAGE_SIZE

# ============================================================================
# FENCE Model Backbone (MSC + Spectral Norm)
# ============================================================================
FENCE_BACKBONE =  "vit_large_patch14_224" #"vit_large_patch14_224"  # or "convnext_v2_large"
FENCE_EMBEDDING_DIM = 768  #768  # ViT-L/14 default, 1536 for ConvNeXt V2-Large
USE_AMP = True

# Mean-Shifted Contrastive (MSC) Loss
MSC_TEMPERATURE = 0.1
MSC_MEAN_SHIFT = True  # Use mean-shifted version
MSC_LAMBDA = 0.5  # Weight for contrastive loss vs classification loss

# Spectral Normalization (Lipschitz constraint)
SPECTRAL_NORM_ENABLED = False  # Apply spectral_norm during training
SPECTRAL_NORM_ITERATIONS = 1  # Power iterations for spectral norm
# FPR loss temperature (smooth sigmoid)
FPR_TEMPERATURE = 10.0
# Input normalization for CLIP (set to True if dataloader already applies CLIP norm)
INPUT_NORMALIZED = False

# ============================================================================
# FENCE Feasible Learning (Constrained Optimization)
# ============================================================================
# Per-sample feasibility tolerance
FL_EPSILON = 0.01  # How much violation is allowed (squared distance)

# Lagrange multiplier update
FL_LR_MULTIPLIER = 0.01  # Learning rate for dual ascent
FL_PI_P = 0.1  # PI controller proportional gain
FL_PI_I = 0.01  # PI controller integral gain

# Resilient slack (for handling outliers/mislabels)
FL_SLACK_WEIGHT = 1.0  # Weight for slack in objective
FL_SLACK_ENABLED = True  # Enable resilient slack

# Optimization
FL_MAX_EPOCHS = 20
FL_CLUSTER_UPDATE_FREQ = 1  # Recompute assignments every N epochs
FL_RADIUS_UPDATE_FREQ = 1  # Recompute radii every N epochs

# ============================================================================
# FENCE Union of Balls (Enclosure Geometry)
# ============================================================================
# Number of balls (k) - set to 0 for auto-detection via clustering
FENCE_NUM_CLUSTERS = 0  # 0 = auto-detect via DBSCAN
FENCE_MAX_CLUSTERS = 10  # Maximum balls if auto-detecting
FENCE_MIN_CLUSTER_SIZE = 5  # Minimum points per cluster

# Radius setting
FENCE_RADIUS_METHOD = 'max'  # 'max' or 'quantile'
FENCE_RADIUS_QUANTILE = 0.95  # If using quantile method
FENCE_RADIUS_MARGIN = 0.1  # Unified margin m (constraint + shell + certified radius)

# Abstention shell (width = FENCE_RADIUS_MARGIN by design)
FENCE_ABSTENTION_ENABLED = True

# ============================================================================
# FENCE OOD Detection (k-NN based)
# ============================================================================
FENCE_OOD_K_NEIGHBORS = 5  # k for k-NN distance
FENCE_OOD_THRESHOLD_QUANTILE = 0.99  # Threshold from training distribution percentile

# ============================================================================
# FENCE Training
# ============================================================================
FENCE_BATCH_SIZE = 64
FENCE_LEARNING_RATE = 1e-6
FENCE_WEIGHT_DECAY = 0.01
FENCE_NUM_EPOCHS = 15

# Validation frequency
FENCE_EVAL_EVERY_N_EPOCHS = 1

# ============================================================================
# FENCE Paths
# ============================================================================
FENCE_CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "fence" / DATASET_NAME / FENCE_BACKBONE
FENCE_LOG_DIR = BASE_DIR / "logs" / "fence"

# ============================================================================
# Create Directories
# ============================================================================
FENCE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FENCE_LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FENCE Two-Radius System
# ============================================================================
FENCE_RADIUS_QUANTILE = 0.9  # Quantile for training radius (creates pressure)
FENCE_DEPLOYMENT_RADIUS_METHOD = 'max'  # 'max' for deployment (guarantees coverage)

# ============================================================================
# Stage 1: MSC Pretraining
# ============================================================================
STAGE1_ENABLED = True  # Whether to run Stage 1 before Stage 2
STAGE1_EPOCHS = 50
STAGE1_LEARNING_RATE = 1e-4
STAGE1_COMPACTNESS_BETA = 0.5  # Weight for compactness term

# ============================================================================
# Stage 2: Feasible Learning
# ============================================================================
STAGE2_LEARNING_RATE = 1e-5  # Lower LR for fine-tuning
STAGE2_FPR_WEIGHT = 0.5  # Weight for FPR loss
STAGE2_GRADIENT_CLIP = 1.0  # Max gradient norm

# ============================================================================
# Finalization
# ============================================================================
FENCE_QUARANTINE_QUANTILE = 0.99  # Top % of slack to quarantine
FENCE_VERIFY_EVERY_EPOCH = 5  # Run verification every N epochs

def print_fence_config():
    """Print FENCE-specific configuration."""
    print("\n" + "="*60)
    print("FENCE CONFIGURATION")
    print("="*60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Backbone: {FENCE_BACKBONE}")
    print(f"Embedding Dim: {FENCE_EMBEDDING_DIM}")
    print()
    print("--- Mean-Shifted Contrastive ---")
    print(f"  Temperature: {MSC_TEMPERATURE}")
    print(f"  Mean Shift: {MSC_MEAN_SHIFT}")
    print(f"  Lambda: {MSC_LAMBDA}")
    print()
    print("--- Spectral Normalization ---")
    print(f"  Enabled: {SPECTRAL_NORM_ENABLED}")
    print(f"  Iterations: {SPECTRAL_NORM_ITERATIONS}")
    print()
    print("--- Feasible Learning ---")
    print(f"  Epsilon: {FL_EPSILON}")
    print(f"  LR Multiplier: {FL_LR_MULTIPLIER}")
    print(f"  PI (P, I): ({FL_PI_P}, {FL_PI_I})")
    print(f"  Slack Enabled: {FL_SLACK_ENABLED}")
    print(f"  Slack Weight: {FL_SLACK_WEIGHT}")
    print(f"  Max Epochs: {FL_MAX_EPOCHS}")
    print()
    print("--- Union of Balls ---")
    print(f"  Num Clusters: {FENCE_NUM_CLUSTERS} (0=auto)")
    print(f"  Max Clusters: {FENCE_MAX_CLUSTERS}")
    print(f"  Radius Method: {FENCE_RADIUS_METHOD}")
    print(f"  Radius Margin m: {FENCE_RADIUS_MARGIN}")
    print(f"  Abstention: {FENCE_ABSTENTION_ENABLED}")
    print()
    print("--- OOD Detection ---")
    print(f"  k-NN: {FENCE_OOD_K_NEIGHBORS}")
    print(f"  Threshold Quantile: {FENCE_OOD_THRESHOLD_QUANTILE}")
    print()
    print("--- Training ---")
    print(f"  Batch Size: {FENCE_BATCH_SIZE}")
    print(f"  Learning Rate: {FENCE_LEARNING_RATE}")
    print(f"  Weight Decay: {FENCE_WEIGHT_DECAY}")
    print(f"  Epochs: {FENCE_NUM_EPOCHS}")
    print(f"  Device: {DEVICE}")
    print(f"  GPUs: {GPU_IDS}")
    print()
    print("--- Paths ---")
    print(f"  Checkpoint: {FENCE_CHECKPOINT_DIR}")
    print(f"  Log: {FENCE_LOG_DIR}")
    print("="*60)


def validate_config():
    """Validate FENCE configuration."""
    print("\nValidating FENCE configuration...")
    
    errors = []
    warnings = []
    
    # Check margin consistency
    if FENCE_RADIUS_MARGIN <= 0:
        errors.append("FENCE_RADIUS_MARGIN must be > 0")
    
    # Check cluster settings
    if FENCE_NUM_CLUSTERS < 0:
        errors.append("FENCE_NUM_CLUSTERS must be >= 0 (0 = auto)")
    
    if FENCE_MAX_CLUSTERS < 1:
        errors.append("FENCE_MAX_CLUSTERS must be >= 1")
    
    # Check radius method
    if FENCE_RADIUS_METHOD not in ['max', 'quantile']:
        errors.append(f"FENCE_RADIUS_METHOD must be 'max' or 'quantile', got {FENCE_RADIUS_METHOD}")
    
    if FENCE_RADIUS_METHOD == 'quantile' and not (0 < FENCE_RADIUS_QUANTILE < 1):
        errors.append("FENCE_RADIUS_QUANTILE must be in (0, 1)")
    
    # Check OOD settings
    if FENCE_OOD_K_NEIGHBORS < 1:
        errors.append("FENCE_OOD_K_NEIGHBORS must be >= 1")
    
    if not (0 < FENCE_OOD_THRESHOLD_QUANTILE < 1):
        errors.append("FENCE_OOD_THRESHOLD_QUANTILE must be in (0, 1)")
    
    # Check feasible learning
    if FL_EPSILON < 0:
        errors.append("FL_EPSILON must be >= 0")
    
    if FL_LR_MULTIPLIER <= 0:
        errors.append("FL_LR_MULTIPLIER must be > 0")
    
    # Warnings
    if FENCE_NUM_CLUSTERS == 0:
        warnings.append("FENCE_NUM_CLUSTERS=0: Auto-detection will be used (may be slow)")
    
    if FENCE_BATCH_SIZE > 128:
        warnings.append(f"Large batch size ({FENCE_BATCH_SIZE}) may cause memory issues")
    
    if errors:
        print("\n❌ Configuration errors:")
        for err in errors:
            print(f"  ✗ {err}")
        return False
    
    if warnings:
        print("\n⚠️ Configuration warnings:")
        for warn in warnings:
            print(f"  ! {warn}")
    
    print("\n✅ Configuration validation passed!")
    return True


# ============================================================================
# Test Cases
# ============================================================================

def test_config_imports():
    """Test that all config values import correctly."""
    print("\n" + "="*60)
    print("TEST: Config Imports")
    print("="*60)
    
    # Check required attributes exist
    required_attrs = [
        'FENCE_BACKBONE', 'FENCE_EMBEDDING_DIM',
        'MSC_TEMPERATURE', 'MSC_MEAN_SHIFT', 'MSC_LAMBDA',
        'SPECTRAL_NORM_ENABLED', 'SPECTRAL_NORM_ITERATIONS',
        'FL_EPSILON', 'FL_LR_MULTIPLIER', 'FL_PI_P', 'FL_PI_I',
        'FL_SLACK_ENABLED', 'FL_SLACK_WEIGHT', 'FL_MAX_EPOCHS',
        'FENCE_NUM_CLUSTERS', 'FENCE_MAX_CLUSTERS', 'FENCE_MIN_CLUSTER_SIZE',
        'FENCE_RADIUS_METHOD', 'FENCE_RADIUS_QUANTILE', 'FENCE_RADIUS_MARGIN',
        'FENCE_ABSTENTION_ENABLED',
        'FENCE_OOD_K_NEIGHBORS', 'FENCE_OOD_THRESHOLD_QUANTILE',
        'FENCE_BATCH_SIZE', 'FENCE_LEARNING_RATE', 'FENCE_WEIGHT_DECAY',
        'FENCE_NUM_EPOCHS', 'FENCE_EVAL_EVERY_N_EPOCHS',
        'FENCE_CHECKPOINT_DIR', 'FENCE_LOG_DIR'
    ]
    
    for attr in required_attrs:
        assert attr in globals(), f"Missing config attribute: {attr}"
        print(f"  ✓ {attr} = {globals()[attr]}")
    
    print("\n✅ All config imports passed!")


def test_device_inheritance():
    """Test that device settings inherit correctly."""
    print("\n" + "="*60)
    print("TEST: Device Inheritance")
    print("="*60)
    
    assert DEVICE == root_config.DEVICE, f"DEVICE mismatch: {DEVICE} vs {root_config.DEVICE}"
    assert GPU_IDS == root_config.GPU_IDS, f"GPU_IDS mismatch: {GPU_IDS} vs {root_config.GPU_IDS}"
    
    print(f"  ✓ DEVICE = {DEVICE}")
    print(f"  ✓ GPU_IDS = {GPU_IDS}")
    
    print("\n✅ Device inheritance passed!")


def test_path_inheritance():
    """Test that paths inherit correctly."""
    print("\n" + "="*60)
    print("TEST: Path Inheritance")
    print("="*60)
    
    assert BASE_DIR == root_config.BASE_DIR, "BASE_DIR mismatch"
    print(f"  ✓ BASE_DIR = {BASE_DIR}")
    
    print(f"  ✓ IMAGE_DIR = {IMAGE_DIR}")
    print(f"  ✓ ANNOTATION_DIR = {ANNOTATION_DIR}")
    
    # Check checkpoint directory creation
    assert FENCE_CHECKPOINT_DIR.exists(), f"Checkpoint dir not created: {FENCE_CHECKPOINT_DIR}"
    assert FENCE_LOG_DIR.exists(), f"Log dir not created: {FENCE_LOG_DIR}"
    
    print(f"  ✓ FENCE_CHECKPOINT_DIR = {FENCE_CHECKPOINT_DIR}")
    print(f"  ✓ FENCE_LOG_DIR = {FENCE_LOG_DIR}")
    
    print("\n✅ Path inheritance passed!")


def run_all_tests():
    """Run all config tests."""
    print("\n" + "="*60)
    print("RUNNING ALL CONFIG TESTS")
    print("="*60)
    
    test_config_imports()
    test_device_inheritance()
    test_path_inheritance()
    
    # Validate configuration
    validate_config()
    
    # Print full config
    print_fence_config()
    
    print("\n" + "="*60)
    print("🎉 ALL CONFIG TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()