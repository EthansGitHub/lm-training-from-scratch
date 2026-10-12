import modal

project_name = "lm-pretraining-owt-base-h100"

app = modal.App(project_name)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--no-dev")
    .add_local_python_source("src")
)

data_vol = modal.Volume.from_name("lm-pretraining-data")
checkpoints_vol = modal.Volume.from_name("lm-pretraining-checkpoints")


@app.function(
    image=image,
    gpu="H100!",
    memory=8192,
    timeout=1 * 60 * 60,
    secrets=[modal.Secret.from_name("wandb")],
    volumes={"/root/data": data_vol, "/root/checkpoints": checkpoints_vol},
    name=project_name,
)
def train_one(
    suffix: int,
    num_train_iters: int,
    batch_size: int,
    num_validate_points: int,
):
    from src.lm_training.experiment import experiment

    experiment(
        project_name=project_name,
        use_wandb=True,
        run_name=f"set0-{suffix}",
        wandb_tag="set0_batch_size",
        checkpoint=False,
        dataset="OWT",
        vocab_size=32000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        device="cuda",
        num_train_iters=num_train_iters,
        num_console_log_iters=25,
        num_validate_iters=50,
        batch_size=batch_size,
        num_validate_points=num_validate_points,
        max_lr=0.004,
        warmup_iters=150,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
    )
    checkpoints_vol.commit()


@app.local_entrypoint()
def train_multiple():
    for suffix, num_train_iters, batch_size, num_validate_points in (
        (0, 600, 256, 3),
        (1, 1200, 128, 6),
        (2, 2400, 64, 12),
    ):
        train_one.spawn(suffix, num_train_iters, batch_size, num_validate_points)
