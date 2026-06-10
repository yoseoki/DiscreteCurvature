"""Estimate top-n Hessian eigenvalues of a trained model on its training data."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from ml_research.analysis.hessian import (
	build_eval_loader,
	load_run,
	top_eigenvalues_batch,
	top_eigenvalues_full,
)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--run-dir", required=True, type=Path,
				   help="Training output directory containing config.yaml and model.pt")
	p.add_argument("--top-n", type=int, default=10)
	p.add_argument("--mode", choices=["batch", "full", "both"], default="both")
	p.add_argument("--batch-index", type=int, default=0,
				   help="Which mini-batch to use in 'batch' mode (shuffle=False loader)")
	p.add_argument("--max-iter", type=int, default=500,
				   help="Power-iteration max steps inside PyHessian")
	p.add_argument("--tol", type=float, default=1e-3)
	p.add_argument("--num-workers", type=int, default=2)
	p.add_argument("--output", type=Path, default=None,
				   help="Output JSON path (default: <run-dir>/hessian.json)")
	return p.parse_args()


def main() -> None:
	args = parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[hessian] device={device}")

	print(f"[hessian] loading run from {args.run_dir}")
	model, trainer, cfg = load_run(args.run_dir, device)
	criterion = trainer.load_criterion().to(device)

	total_norm = sum(p.norm()**2 for p in model.parameters())**0.5
	print("total norm : {}".format(total_norm))
	# for name, p in model.named_parameters():
	# 	print(name, p.norm().item())

	print(f"[hessian] building shuffle=False train loader (batch_size={trainer.batch_size})")
	loader = build_eval_loader(trainer, num_workers=args.num_workers)
	n_batches = len(loader)

	results: dict[str, dict] = {}

	if args.mode in ("batch", "both"):
		print(f"[hessian] mode=batch (index={args.batch_index}) top_n={args.top_n}")
		t0 = time.perf_counter()
		eigvals = top_eigenvalues_batch(
			model, criterion, loader,
			top_n=args.top_n, device=device,
			batch_index=args.batch_index,
			max_iter=args.max_iter, tol=args.tol,
		)
		elapsed = time.perf_counter() - t0
		print(f"[hessian] batch eigenvalues ({elapsed:.1f}s): {eigvals}")
		results["batch"] = {
			"eigenvalues": eigvals,
			"batch_index": args.batch_index,
			"batch_size": trainer.batch_size,
			"max_iter": args.max_iter,
			"tol": args.tol,
			"elapsed_sec": elapsed,
		}

	if args.mode in ("full", "both"):
		print(f"[hessian] mode=full n_batches={n_batches} top_n={args.top_n} (this may take a while)")
		t0 = time.perf_counter()
		eigvals = top_eigenvalues_full(
			model, criterion, loader,
			top_n=args.top_n, device=device,
			max_iter=args.max_iter, tol=args.tol,
		)
		elapsed = time.perf_counter() - t0
		print(f"[hessian] full eigenvalues ({elapsed:.1f}s): {eigvals}")
		results["full"] = {
			"eigenvalues": eigvals,
			"n_batches": n_batches,
			"batch_size": trainer.batch_size,
			"max_iter": args.max_iter,
			"tol": args.tol,
			"elapsed_sec": elapsed,
		}

	payload = {
		"run_dir": str(args.run_dir.resolve()),
		"model_name": trainer.model_name,
		"model_backend": trainer.model_backend,
		"data_name": cfg.data.name,
		"seed": trainer.seed,
		"top_n": args.top_n,
		"device": str(device),
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"results": results,
	}

	out_path = args.output if args.output is not None else (args.run_dir / "hessian.json")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "w") as f:
		json.dump(payload, f, indent=2)
	print(f"[hessian] wrote {out_path}")


if __name__ == "__main__":
	main()
