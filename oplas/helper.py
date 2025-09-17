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



def get_scheduler(opt, config):
    scheduler = None

    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=config.max_steps
        )

    return scheduler
