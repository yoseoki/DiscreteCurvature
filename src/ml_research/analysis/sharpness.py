"""Compute rho-sharpness (SAM-style) around a trained model.

rho-sharpness measures how much the loss increases when the weights are
perturbed by epsilon = rho * g/||g|| along the (full-batch or mini-batch)
gradient ascent direction:

    sharpness(rho) = L(w + rho * g/||g||) - L(w),   g = grad_w L(w)

Larger value => sharper minimum.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

# Reuse run reconstruction / loader helpers from the Hessian module so the two
# analyses operate on identical model weights and data ordering.
from ml_research.analysis.hessian import (  # noqa: F401  (re-exported for callers)
    build_eval_loader,
    load_run,
    _get_nth_batch,
)


def _grad_global_norm(grads: list[torch.Tensor]) -> torch.Tensor:
    """Global L2 norm of the concatenated gradient vector."""
    return torch.norm(torch.stack([g.norm() for g in grads]))


def _sharpness_from_grads(
    model: torch.nn.Module,
    eval_loss_fn,
    params: list[torch.nn.Parameter],
    grads: list[torch.Tensor],
    loss_clean: float,
    rhos: list[float],
) -> dict[float, float]:
    """Given the gradient at w, measure L(w + eps) - L(w) for each rho.

    The weights are snapshotted and restored exactly, so multiple rho values
    reuse the same (expensive) gradient computation.
    """
    grad_norm = _grad_global_norm(grads)
    originals = [p.detach().clone() for p in params]

    out: dict[float, float] = {}
    with torch.no_grad():
        for rho in rhos:
            scale = rho / (grad_norm + 1e-12)
            for p, g in zip(params, grads):
                p.add_(g * scale)
            loss_pert = eval_loss_fn()
            # restore exactly from the snapshot (avoids float accumulation drift)
            for p, orig in zip(params, originals):
                p.copy_(orig)
            out[float(rho)] = float(loss_pert - loss_clean)
    return out


def rho_sharpness_batch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    rhos: list[float],
    device: torch.device,
    batch_index: int = 0,
) -> dict:
    """rho-sharpness on a single fixed mini-batch."""
    X, Y = _get_nth_batch(loader, batch_index)
    X = X.to(device)
    Y = Y.to(device)

    model.zero_grad(set_to_none=True)
    loss = criterion(model(X), Y)
    loss_clean = float(loss.item())
    loss.backward()

    params = [p for p in model.parameters() if p.grad is not None]
    grads = [p.grad.detach().clone() for p in params]

    def eval_loss_fn() -> float:
        return float(criterion(model(X), Y).item())

    sharpness = _sharpness_from_grads(
        model, eval_loss_fn, params, grads, loss_clean, rhos
    )
    model.zero_grad(set_to_none=True)
    return {
        "loss_clean": loss_clean,
        "grad_norm": float(_grad_global_norm(grads).item()),
        "sharpness": sharpness,
        "batch_index": batch_index,
    }


def rho_sharpness_full(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    rhos: list[float],
    device: torch.device,
) -> dict:
    """rho-sharpness using the full-dataset gradient and loss.

    Both the gradient g = grad_w L(w) and the clean/perturbed losses are
    sample-weighted averages over the entire loader, matching the standard
    full-batch definition (mean-reduction criterion assumed).
    """
    model.zero_grad(set_to_none=True)

    # Accumulate sample-weighted gradient and clean loss over the whole dataset.
    total_n = 0
    loss_clean_sum = 0.0
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        n = X.shape[0]
        loss = criterion(model(X), Y)
        # weight each batch's mean loss by its sample count, normalize later
        (loss * n).backward()
        loss_clean_sum += float(loss.item()) * n
        total_n += n

    params = [p for p in model.parameters() if p.grad is not None]
    # grads currently hold sum_b (n_b * grad of mean-loss_b); divide by N.
    grads = [(p.grad.detach() / total_n).clone() for p in params]
    loss_clean = loss_clean_sum / total_n

    def eval_loss_fn() -> float:
        s = 0.0
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            n = X.shape[0]
            s += float(criterion(model(X), Y).item()) * n
        return s / total_n

    sharpness = _sharpness_from_grads(
        model, eval_loss_fn, params, grads, loss_clean, rhos
    )
    model.zero_grad(set_to_none=True)
    return {
        "loss_clean": loss_clean,
        "grad_norm": float(_grad_global_norm(grads).item()),
        "sharpness": sharpness,
        "n_samples": total_n,
    }
