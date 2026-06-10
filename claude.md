# Project: ML Research Workspace

## Purpose
CV/NLP research on image classification and representation learning with
ResNet, ViT, VGG, and related backbones.

## Stack
- **Env**: pixi (see `pixi.toml`)
- **Config**: Hydra (`conf/`)
- **Training**: PyTorch + timm (+ Huggingface Accelerate (Not implemented))
- **Logging**: Weights & Biases
- **Metrics**: (torchmetrics (Not implemented))
- **Docs**: Quarto (`paper/`, `notebooks/`)

## Directory conventions
- `src/` — library code only, no scripts, no `if __name__ == "__main__"`.
- `scripts/` — entrypoints (`train.py`, `validate_geodesic.py`, `eval.py`). Thin wrappers over `src/`.
- `conf/` — Hydra configs. Structured as `model/`, `data/`, `experiment/`.
  - Adding a new model means: (1) new file in `src/models/`, (2) new config in
    `conf/model/<name>.yaml`, (3) registry entry in `src/models/__init__.py`.
- `notebooks/` — `.qmd` files for analysis. These read from `outputs/` and
  write figures to `paper/figures/`. Never hardcode numbers in the paper.
- `paper/` — Quarto paper project. Figures and tables come from `notebooks/`.
- `outputs/` — Hydra run artifacts. Gitignored. Each run has its own dir.
- `tests/` — pytest. Shape tests for models, smoke tests for training.

## Coding conventions
- Type hints everywhere. Use `jaxtyping` for tensor shapes:
  `Float[Tensor, "batch channel height width"]`.
- No magic numbers in code — put them in Hydra config.
- `einops.rearrange` instead of `.view` / `.permute` when reshaping.
- Seeds set via `accelerate.utils.set_seed(cfg.seed)` at entry of every script.
- Every new module gets a shape test in `tests/`.

## Running experiments
```bash
pixi run train experiment=resnet50_cifar100
pixi run sweep experiment=vit_cifar100 optimizer.lr=1e-4,3e-4,1e-3
pixi run eval run_dir=outputs/2026-04-08/12-00-00
pixi run report     # regenerates analysis + paper
```

## Workflow for Claude Code
When asked to add a new experiment:
1. Check if model/data configs already exist in `conf/`. Reuse if possible.
2. Create `conf/experiment/<name>.yaml` composing existing pieces.
3. Add a smoke test in `tests/test_experiments.py` that runs 2 steps.
4. Do NOT start a real training run — tell me the command to run.

When asked to add a new model:
1. Prefer `timm.create_model(...)` over reimplementing.
2. New file in `src/models/<name>.py` with a `build(cfg) -> nn.Module` function.
3. Register in `src/models/__init__.py`.
4. Add config in `conf/model/<name>.yaml`.
5. Add a shape test in `tests/test_models.py`.

When asked to analyze results:
1. Edit `notebooks/analysis.qmd`, not the paper directly.
2. Load runs from `outputs/` or W&B API.
3. Save figures to `paper/figures/` with descriptive names.
4. The paper `.qmd` should only reference them via `![](figures/...)`.

## Things to NOT do
- Don't commit to `main` without running `pixi run check`.
- Don't put numbers in the paper manually — they go through `analysis.qmd`.
- Don't add dependencies without updating `pixi.toml`.
- Don't create new top-level directories without asking.
