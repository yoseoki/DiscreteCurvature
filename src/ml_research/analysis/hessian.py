"""Estimate top eigenvalues of the loss-Hessian around a trained model."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ml_research.training import trainer as trainer_mod


_TRAINER_REGISTRY: dict[str, type] = {
	"cifar10": trainer_mod.CIFAR10Trainer,
	"cifar100": trainer_mod.CIFAR100Trainer,
	"svhn": trainer_mod.SVHNTrainer,
}


def load_run(run_dir: str | os.PathLike, device: torch.device):
	"""Reconstruct (model, trainer, cfg) from a finished training run directory."""
	run_dir = Path(run_dir)
	cfg_path = run_dir / "config.yaml"
	weight_path = run_dir / "model.pt"
	if not cfg_path.is_file():
		raise FileNotFoundError(f"config.yaml not found in {run_dir}")
	if not weight_path.is_file():
		raise FileNotFoundError(f"model.pt not found in {run_dir}")

	cfg = OmegaConf.load(cfg_path)

	data_name = str(cfg.data.name).lower()
	if data_name not in _TRAINER_REGISTRY:
		raise ValueError(
			f"Unsupported data.name={data_name!r} (known: {list(_TRAINER_REGISTRY)})"
		)
	trainer = _TRAINER_REGISTRY[data_name]()
	trainer.parse_training_args(cfg)
	trainer.prefix_w = str(run_dir)
	trainer.set_seed()

	model = trainer.load_model().to(device)
	state = torch.load(weight_path, map_location=device, weights_only=True)
	model.load_state_dict(state)
	model.eval()

	return model, trainer, cfg


def build_eval_loader(
	trainer,
	batch_size: int | None = None,
	num_workers: int = 2,
) -> DataLoader:
	"""Rebuild the train loader with shuffle=False so batch-mode Hessian is reproducible."""
	train_loader, _ = trainer.load_DB()
	dataset = train_loader.dataset
	bs = batch_size if batch_size is not None else train_loader.batch_size
	return DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers)


def _get_nth_batch(loader: DataLoader, index: int):
	it = iter(loader)
	for _ in range(index):
		next(it)
	return next(it)


def top_eigenvalues_batch(
	model: torch.nn.Module,
	criterion: torch.nn.Module,
	loader: DataLoader,
	top_n: int,
	device: torch.device,
	batch_index: int = 0,
	max_iter: int = 100,
	tol: float = 1e-3,
) -> list[float]:
	"""Top-n Hessian eigenvalues on a single fixed mini-batch."""
	from pyhessian import hessian

	X, Y = _get_nth_batch(loader, batch_index)
	X = X.to(device)
	Y = Y.to(device)

	H = hessian(model, criterion, data=(X, Y), cuda=(device.type == "cuda"))
	eigvals, _ = H.eigenvalues(maxIter=max_iter, tol=tol, top_n=top_n)
	return [float(v) for v in eigvals]


def top_eigenvalues_full(
	model: torch.nn.Module,
	criterion: torch.nn.Module,
	loader: DataLoader,
	top_n: int,
	device: torch.device,
	max_iter: int = 100,
	tol: float = 1e-3,
) -> list[float]:
	"""Top-n Hessian eigenvalues averaged over the entire dataloader."""
	from pyhessian import hessian

	H = hessian(model, criterion, dataloader=loader, cuda=(device.type == "cuda"))
	eigvals, _ = H.eigenvalues(maxIter=max_iter, tol=tol, top_n=top_n)
	return [float(v) for v in eigvals]
