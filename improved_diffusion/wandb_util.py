import os
import torch
import wandb
import re


def get_available_steps(checkpoints):
    matches = [re.search(r'(\d+).pt', checkpoint) for checkpoint in checkpoints]
    return list(sorted(set(m.group(1) for m in matches if m)))


def get_ckpt_type(ckpt_type, checkpoints):
    return [ch for ch in checkpoints if ckpt_type in ch][0]


def download_checkpoints(run_id, step=-1, checkpoints=None):
    if checkpoints is None:
        checkpoints = list_all_checkpoints(run_id)

    if step == -1:
        step = max(get_available_steps(checkpoints))

    downloaded_checkpoints = []
    for checkpoint in checkpoints:
        if str(step) in checkpoint:
            download_checkpoint(run_id, checkpoint)
            downloaded_checkpoints.append(checkpoint)

    return {
        ckpt_type: f"./data/{run_id}/" + get_ckpt_type(ckpt_type, downloaded_checkpoints)
        for ckpt_type in ['ema', 'model', 'opt']
    }


def list_all_checkpoints(run_id, project="ddpm/diffusion"):
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    files = run.files()
    checkpoints = []
    for file in files:
        if file.name.endswith(".pt"):
            checkpoints.append(file.name)
    return checkpoints


def download_checkpoint(run_id, checkpoint, project="ddpm/diffusion"):
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    files = run.files()
    for file in files:
        if file.name == checkpoint:
            file.download(f"./data/{run_id}/", replace=True)
            return

def download_samples(run_id, project="ddpm/diffusion"):
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    files = run.files()
    for file in files:
        if "samples" in file.name:
            file.download(f"./data/{run_id}/", replace=True)
            return f"./data/{run_id}/{file.name}"

