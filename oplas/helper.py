from pathlib import Path
import torch


def save_model(
    model, step, optimizer, scheduler, loss, run_id, save_dir="./checkpoints"
):
    """Saves the model checkpoint in a run-specific directory."""
    run_dir = Path(save_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / f"checkpoint_step_{step}.pt"

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
            if scheduler
            else None,  # Save scheduler state
            "loss": loss,
        },
        save_path,
    )
    return save_path


def get_scheduler(opt, config, dl):
    scheduler = None

    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=config.max_steps
        )
    if config.scheduler == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            opt, base_lr=config.base_lr, max_lr=config.max_lr
        )
    if config.scheduler == "cyclic2":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            opt, base_lr=config.base_lr, max_lr=config.max_lr, mode="triangular2"
        )
    if config.scheduler == "1cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=config.max_lr, epochs=config.max_epochs, steps_per_epoch=len(dl),
        )
    if config.scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 'min', patience=3,
        )
    return scheduler
