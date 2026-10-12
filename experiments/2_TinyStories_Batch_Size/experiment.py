import modal

project_name = "lm-pretraining-tiny-stories-batch-size"

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
    gpu="A100-80GB",  # or "A100-40GB"
    memory=8192,
    timeout=3 * 60 * 60,
    secrets=[modal.Secret.from_name("wandb")],
    volumes={"/root/data": data_vol, "/root/checkpoints": checkpoints_vol},
    name=project_name,
)
def train_one(suffix: int, batch_size: int, num_validate_points: int, num_train_iters: int, max_lr: float):
    from src.lm_training.experiment import experiment

    experiment(
        project_name=project_name,
        use_wandb=True,
        run_name=f"set0-{suffix}",
        wandb_tag="set0_start",
        checkpoint=False,
        dataset="TinyStories",
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        device="cuda",
        num_train_iters=num_train_iters,
        num_console_log_iters=10,
        num_validate_iters=25,
        batch_size=batch_size,
        num_validate_points=num_validate_points,
        max_lr=max_lr,
        warmup_iters=150,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
    )
    checkpoints_vol.commit()


@app.local_entrypoint()
def train_multiple():
    for suffix, batch_size, num_validate_points, num_train_iters, max_lr in (
        (0, 768, 1, 200, 0.008),
        (3, 512, 2, 300, 0.008),
        (19, 256, 3, 600, 0.007),
        (32, 128, 6, 1200, 0.005),
        (36, 64, 12, 2400, 0.003),
        (40, 32, 24, 4800, 0.002),
        (48, 8, 96, 19200, 0.001),
        (58, 1, 768, 19200, 0.0006),
    ):
        train_one.spawn(suffix, batch_size, num_validate_points, num_train_iters, max_lr)
