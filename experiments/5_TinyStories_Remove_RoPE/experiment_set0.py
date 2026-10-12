import modal

project_name = "lm-pretraining-tiny-stories-remove-rope"

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
def train_one(suffix: int, max_lr: float):
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
        num_train_iters=200,
        num_console_log_iters=20,
        num_validate_iters=50,
        batch_size=768,
        num_validate_points=1,
        max_lr=max_lr,
        warmup_iters=150,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
    )
    checkpoints_vol.commit()


@app.local_entrypoint()
def train_multiple():
    for suffix, max_lr in (
        (0, 0.008),
        (1, 0.01),
        (2, 0.03),
        (3, 0.003),
        (4, 0.001),
        (5, 0.009),
        (6, 0.006),
        (7, 0.004),
        (8, 0.005),
        (9, 0.007),
    ):
        train_one.spawn(suffix, max_lr)
