# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TTT-Video-Gen is a research project that extends **CogVideoX-5B** (a 5B diffusion transformer) for long-range video generation (up to 63 seconds) using **Test-Time Training (TTT) layers**. The core idea: TTT layers handle global context across 3-second segments while original pretrained attention layers handle local attention within each segment.

## Setup

```bash
# Create environment
conda env create -f environment.yaml && conda activate ttt-video

# Install TTT-MLP CUDA kernels (requires CUDA 12.3+, GCC 11+)
git submodule update --init --recursive
(cd ttt-tk && python setup.py install)

# Install package
pip install -e .

# Convert CogVideoX pretrained weights from HuggingFace
bash scripts/convert_weights_from_hf.sh
```

## Common Commands

**Training (single node, 8 GPUs):**
```bash
torchrun --nproc_per_node=8 train.py \
  --job.config_file=configs/train/ttt-mlp/3s.toml \
  --checkpoint.init_state_dir=<MODEL_WEIGHTS_DIR> \
  --training.dataset_path=<DATA_DIR> \
  --training.jsonl_paths=<METADATA_JSONL> \
  --parallelism.dp_replicate=1 \
  --parallelism.dp_sharding=8 \
  --parallelism.tp_sharding=1
```

**Sampling/Inference:**
```bash
torchrun --nproc_per_node=8 sample.py \
  --job.config_file=configs/eval/ttt-mlp/9s.toml \
  --checkpoint.init_state_dir=<CHECKPOINT_DIR> \
  --eval.input_file=inputs/example-9s.json \
  --eval.output_dir=./output
```

**Dataset preparation:**
```bash
python data/precomp_video.py   # VAE-encode video segments
python data/precomp_text.py    # T5-embed text descriptions
```

**Code formatting/linting:**
```bash
black .
flake8 .
isort .
```

## Architecture

### Core Flow

```
Text Prompts → T5 Encoder → Text Embeddings
                                    ↓
Video Latents ← VAE Decode ← Diffusion Transformer (DIT)
                                    ↑
                          TTT Layers (global)
                        + Local Attention (3s segments)
```

### Key Packages

**`ttt/models/`** — Neural network components:
- `cogvideo/dit.py`: Core DIT architecture with TTT layer injection
- `cogvideo/model.py`: CogVideoX wrapper with loss calculation (L2 with noise weighting)
- `cogvideo/sampler.py`: DDIM sampling pipeline with classifier-free guidance
- `cogvideo/utils.py`: Noise scheduling, guidance utilities
- `ssm/ttt_layer.py`: TTT layer wrapper (supports both TTT-MLP and TTT-Linear variants)
- `ssm/mlp_tk.py`: TTT-MLP with custom CUDA kernels (via `ttt-tk` submodule)
- `ssm/linear_triton.py`: TTT-Linear using Triton kernels
- `configs.py`: `ModelConfig` dataclasses for each video length (3s/9s/18s/30s/63s)

**`ttt/infra/`** — Training infrastructure:
- `config_manager.py`: TOML-based `JobConfig` system (all CLI flags map to nested TOML keys)
- `parallelisms.py`: Distributed training setup — HSDP (Hybrid Sharded DP), FSDP, Tensor Parallelism
- `optimizers.py`: Separate parameter groups for TTT vs. non-TTT params with different LR
- `checkpoint.py`: Save/resume with FSDP-aware state dicts
- `train_iterator.py`: Training loop with gradient accumulation and fault tolerance

**`ttt/datasets/`** — Data loading:
- `preembedding_dataset.py`: Loads precomputed VAE + T5 embeddings from JSONL manifests
- `data_sampler.py`: Fault-tolerant distributed sampler

### Training Curriculum (5 stages)

1. **3s**: Full fine-tuning from CogVideoX pretrained weights
2. **9s → 18s → 30s → 63s**: Progressive stages training only TTT + QKV projection layers

Each stage uses configs in `configs/train/ttt-mlp/` or `configs/train/ttt-linear/`.

### Configuration System

All configuration uses TOML files parsed by `JobConfig` in `ttt/infra/config_manager.py`. CLI flags use dot notation to override TOML values (e.g., `--parallelism.dp_sharding=8`). Top-level config sections: `job`, `model`, `training`, `eval`, `parallelism`, `checkpoint`, `optimizer`, `remat`, `wandb`.

### Dataset Format

Training data is JSONL where each line contains:
```json
{
  "vid_emb": "<path_to_video_latent.pt>",
  "text_chunk_emb": ["<text_emb_scene1.pt>", "<text_emb_scene2.pt>", ...]
}
```

### Hardware Requirements

- TTT-MLP: H100 GPUs required (custom CUDA kernels)
- TTT-Linear: A100+ GPUs (Triton kernels)
- Multi-node training: use `train_submitit.py` + `scripts/train_submitit.sh` (SLURM)
