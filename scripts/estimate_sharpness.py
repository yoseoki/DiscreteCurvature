"""Estimate rho-sharpness (SAM-style) of a trained model on its training data."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from ml_research.analysis.hessian import build_eval_loader, load_run
from ml_research.analysis.sharpness import rho_sharpness_batch, rho_sharpness_full


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--run-dir", required=True, type=Path,
				   help="Training output directory containing config.yaml and model.pt")
	p.add_argument("--rho", type=float, nargs="+", default=[0.05],
				   help="One or more perturbation radii (gradient reused across them)")
	p.add_argument("--mode", choices=["batch", "full", "both"], default="both")
	p.add_argument("--batch-index", type=int, default=0,
				   help="Which mini-batch to use in 'batch' mode (shuffle=False loader)")
	p.add_argument("--num-workers", type=int, default=2)
	p.add_argument("--output", type=Path, default=None,
				   help="Output JSON path (default: <run-dir>/sharpness.json)")
	return p.parse_args()


def main() -> None:
	args = parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[sharpness] device={device}")

	print(f"[sharpness] loading run from {args.run_dir}")
	model, trainer, cfg = load_run(args.run_dir, device)
	criterion = trainer.load_criterion().to(device)

	print(f"[sharpness] building shuffle=False train loader (batch_size={trainer.batch_size})")
	loader = build_eval_loader(trainer, num_workers=args.num_workers)
	n_batches = len(loader)

	results: dict[str, dict] = {}

	if args.mode in ("batch", "both"):
		print(f"[sharpness] mode=batch (index={args.batch_index}) rho={args.rho}")
		t0 = time.perf_counter()
		res = rho_sharpness_batch(
			model, criterion, loader,
			rhos=args.rho, device=device,
			batch_index=args.batch_index,
		)
		elapsed = time.perf_counter() - t0
		print(f"[sharpness] batch ({elapsed:.1f}s): {res['sharpness']}")
		res["batch_size"] = trainer.batch_size
		res["elapsed_sec"] = elapsed
		results["batch"] = res

	if args.mode in ("full", "both"):
		print(f"[sharpness] mode=full n_batches={n_batches} rho={args.rho} (this may take a while)")
		t0 = time.perf_counter()
		res = rho_sharpness_full(
			model, criterion, loader,
			rhos=args.rho, device=device,
		)
		elapsed = time.perf_counter() - t0
		print(f"[sharpness] full ({elapsed:.1f}s): {res['sharpness']}")
		res["n_batches"] = n_batches
		res["batch_size"] = trainer.batch_size
		res["elapsed_sec"] = elapsed
		results["full"] = res

	payload = {
		"run_dir": str(args.run_dir.resolve()),
		"model_name": trainer.model_name,
		"model_backend": trainer.model_backend,
		"data_name": cfg.data.name,
		"seed": trainer.seed,
		"rho": args.rho,
		"device": str(device),
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"results": results,
	}

	out_path = args.output if args.output is not None else (args.run_dir / "sharpness.json")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "w") as f:
		json.dump(payload, f, indent=2)
	print(f"[sharpness] wrote {out_path}")


if __name__ == "__main__":
	main()
