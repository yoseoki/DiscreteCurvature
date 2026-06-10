"""Post-hoc generalization analysis for finished training runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import pandas as pd


_DEFAULT_TARGETS: tuple[float, ...] = (2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02)


def load_metrics(run_dir: str | os.PathLike) -> pd.DataFrame:
	"""Load epoch-aligned metrics; prefer metrics.csv, fall back to four individual CSVs."""
	run_dir = Path(run_dir)
	unified = run_dir / "metrics.csv"
	if unified.is_file():
		return pd.read_csv(unified)

	needed = {
		"train_cost.csv": "train_cost",
		"train_acc.csv":  "train_acc",
		"test_cost.csv":  "test_cost",
		"test_acc.csv":   "test_acc",
	}
	missing = [name for name in needed if not (run_dir / name).is_file()]
	if missing:
		raise FileNotFoundError(
			f"Cannot load metrics from {run_dir}: missing {missing} and no metrics.csv. "
			f"This run was likely trained before the train-acc/test-loss instrumentation."
		)
	df: pd.DataFrame | None = None
	for fname in needed:
		sub = pd.read_csv(run_dir / fname)
		df = sub if df is None else df.merge(sub, on="epoch")
	assert df is not None
	return df


def compute_gen_gap(run_dir: str | os.PathLike) -> pd.DataFrame:
	"""Per-epoch accuracy & loss generalization gap; written to run_dir/gen_gap.csv.

	Sign convention: both gaps are positive when the run overfits.
	  acc_gap  = train_acc  - test_acc   (acc:  train > test  when overfit)
	  loss_gap = test_cost  - train_cost (loss: test  > train when overfit)
	"""
	run_dir = Path(run_dir)
	m = load_metrics(run_dir)
	out = pd.DataFrame({
		"epoch":      m["epoch"],
		"train_cost": m["train_cost"],
		"test_cost":  m["test_cost"],
		"loss_gap":   m["test_cost"] - m["train_cost"],
		"train_acc":  m["train_acc"],
		"test_acc":   m["test_acc"],
		"acc_gap":    m["train_acc"] - m["test_acc"],
	})
	out.to_csv(run_dir / "gen_gap.csv", index=False)
	return out


def _interp_at_target(
	epochs: Sequence[float],
	train_losses: Sequence[float],
	test_losses: Sequence[float],
	test_accs: Sequence[float],
	target: float,
) -> tuple[float, float, float, str]:
	"""Find the first downward crossing of `target` and linearly interpolate."""
	n = len(epochs)
	if n == 0:
		return float("nan"), float("nan"), float("nan"), "empty"
	if train_losses[0] < target:
		return float("nan"), float("nan"), float("nan"), "already_below"
	for i in range(n - 1):
		a, b = float(train_losses[i]), float(train_losses[i + 1])
		if a >= target and b < target:
			denom = a - b
			t = 0.0 if denom == 0.0 else (a - target) / denom
			ep = epochs[i] + t * (epochs[i + 1] - epochs[i])
			tl = test_losses[i] + t * (test_losses[i + 1] - test_losses[i])
			ta = test_accs[i] + t * (test_accs[i + 1] - test_accs[i])
			return float(ep), float(tl), float(ta), "ok"
	return float("nan"), float("nan"), float("nan"), "never_reached"


def match_test_loss_at_train_loss(
	run_dir: str | os.PathLike,
	targets: Sequence[float] = _DEFAULT_TARGETS,
) -> pd.DataFrame:
	"""For each target train loss, linearly interpolate matching test loss/acc.

	Writes run_dir/matched_test_loss.csv. Status column is one of:
	  - "ok":            interpolation succeeded
	  - "already_below": first epoch's train loss already below the target
	  - "never_reached": train loss never falls below the target
	"""
	run_dir = Path(run_dir)
	m = load_metrics(run_dir)
	epochs       = m["epoch"].tolist()
	train_losses = m["train_cost"].tolist()
	test_losses  = m["test_cost"].tolist()
	test_accs    = m["test_acc"].tolist()

	rows = []
	for L in targets:
		ep, tl, ta, status = _interp_at_target(
			epochs, train_losses, test_losses, test_accs, float(L)
		)
		rows.append({
			"target_train_loss": float(L),
			"interp_epoch":      ep,
			"interp_test_loss":  tl,
			"interp_test_acc":   ta,
			"status":            status,
		})
	out = pd.DataFrame(rows)
	out.to_csv(run_dir / "matched_test_loss.csv", index=False)
	return out
