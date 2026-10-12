import gc
import time

import torch

from src.lm_training.lm_trainer import LMTrainer


def experiment(
    project_name: str,
    use_wandb: bool,
    run_name: str,
    wandb_tag: str,
    checkpoint: bool,
    dataset: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    device: str,
    num_train_iters: int,
    num_console_log_iters: int,
    num_validate_iters: int,
    batch_size: int,
    num_validate_points: int,
    max_lr: float,
    warmup_iters: int,
    weight_decay: float,
    beta1: float,
    beta2: float,
):
    print("Run", run_name)

    if device == "cuda":
        print("cuda:", torch.cuda.is_available())

    match dataset:
        case "TinyStories":
            train_data_file_prefix = "TinyStoriesV2-GPT4-train"
            validate_data_file_prefix = "TinyStoriesV2-GPT4-valid"
        case "OWT":
            train_data_file_prefix = "owt_train"
            validate_data_file_prefix = "owt_valid"
        case _:
            raise Exception(f"{dataset} dataset not supported.")

    train_loss_stop = 1000

    default_trainer_config = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "device": device,
    }

    training_config = {
        "dataset": dataset,
        "num_train_iters": num_train_iters,
        "num_console_log_iters": num_console_log_iters,
        "num_validate_iters": num_validate_iters,
        "batch_size": batch_size,
        "num_validate_points": num_validate_points,
    }

    varying_trainer_config = {
        "max_lr": max_lr,
        "min_lr": max_lr * 0.1,
        "warmup_iters": warmup_iters,
        "cosine_cycle_iters": num_train_iters,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
        "gradient_ceiling": 1,
    }

    config = default_trainer_config | varying_trainer_config | training_config

    trainer = LMTrainer(
        **default_trainer_config,
        **varying_trainer_config,
        train_data_file_prefix=train_data_file_prefix,
        validate_data_file_prefix=validate_data_file_prefix,
    )

    if use_wandb:
        import wandb
        wandb.login()

        run = wandb.init(
            project=project_name,
            config=config,
            name=run_name,
            tags=[wandb_tag],
        )

    start = time.perf_counter()
    i = 0
    for i in range(num_train_iters + 1):
        train_loss, grad_norm_pre_clip, lapse = trainer.train_one_iteration(batch_size)

        to_console = i % num_console_log_iters == 0
        early_stop = train_loss >= train_loss_stop

        if to_console or early_stop:
            print(f"Iteration {i}")
            print(f"    Training loss: {train_loss}")
            print(f"    Grad norm pre clip: {grad_norm_pre_clip}")

            total_lapse = time.perf_counter() - start
            print(f"    Total runtime: {total_lapse:.2f}")

            if use_wandb:
                run.log({ # type: ignore
                    "train loss": train_loss,
                    "grad norm": grad_norm_pre_clip,
                    "iter runtime": lapse,
                    "total runtime": total_lapse,
                }, step=i)

        to_validate = num_validate_iters > 0 and i % num_validate_iters == 0
        if to_validate or early_stop:
            if not to_console:
                print(f"Iteration {i}")

            valid_loss = trainer.validate_loss(batch_size, num_validate_points)
            print(f"    Validation loss: {valid_loss}")

            if use_wandb:
                run.log({ # type: ignore
                    "valid loss": valid_loss,
                }, step=i)

        if early_stop:
            print(f"Iteration {i}: Training loss {train_loss} above the threshold {train_loss_stop}. Abort.")
            break

    if use_wandb:
        run.finish() # type: ignore

    if checkpoint:
        trainer.save_checkpoint(f"./checkpoints/checkpoint-{run_name}.pt", include_optimizer=False)

    if device == "mps":
        print(f"mps memory: {torch.mps.driver_allocated_memory() / 1073741824:.2f} GiB")
    if device == "cuda":
        print(f"cuda memory: {torch.cuda.max_memory_allocated() / 1073741824:.2f} GiB")

    del trainer
    gc.collect()
