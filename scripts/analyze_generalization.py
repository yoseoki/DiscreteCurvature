"""Run post-hoc generalization analyses on one or more training runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_research.analysis.generalization import (
	compute_gen_gap,
	match_test_loss_at_train_loss,
)


_DEFAULT_TARGETS: list[float] = [2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--run-dir", required=True, nargs="+", type=Path,
				   help="One or more training output directories (each containing metrics.csv)")
	p.add_argument("--targets", type=str, default=None,
				   help="Comma-separated train loss targets (default: built-in grid)")
	p.add_argument("--skip-gap", action="store_true",
				   help="Skip the generalization-gap step (gen_gap.csv)")
	p.add_argument("--skip-match", action="store_true",
				   help="Skip the matched-train-loss step (matched_test_loss.csv)")
	return p.parse_args()


def _parse_targets(s: str | None) -> list[float]:
	if s is None:
		return _DEFAULT_TARGETS
	return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
	args = parse_args()
	targets = _parse_targets(args.targets)

	for run_dir in args.run_dir:
		print(f"\n[generalization] === {run_dir} ===")
		try:
			if not args.skip_gap:
				gap = compute_gen_gap(run_dir)
				last = gap.iloc[-1]
				print(f"[generalization] wrote {run_dir / 'gen_gap.csv'} ({len(gap)} epochs)")
				print(f"  final epoch: acc_gap={last['acc_gap']:.4f}, loss_gap={last['loss_gap']:.4f}")

			if not args.skip_match:
				matched = match_test_loss_at_train_loss(run_dir, targets)
				ok = int((matched["status"] == "ok").sum())
				print(f"[generalization] wrote {run_dir / 'matched_test_loss.csv'} ({ok}/{len(matched)} ok)")

		except FileNotFoundError as e:
			print(f"[generalization] SKIP {run_dir}: {e}")


if __name__ == "__main__":
	main()
