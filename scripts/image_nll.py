"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os

import numpy as np
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.wandb_util import download_checkpoints

import wandb
import os


def main():
    wandb.init(project='diffusion', entity='ddpm', tags=["openai","nll"], dir="/scratch/diffusion")
    args = create_argparser().parse_args()

    print(vars(args))
    wandb.config.update(args)
    os.environ["OPENAI_LOGDIR"] = wandb.run.dir

    if args.download_checkpoint:
        args.model_path = download_checkpoints(
            args.download_checkpoint,
            args.download_step
        )[args.resume_type]

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    logger.log("evaluating...")
    run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised)


def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised):
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0) / dist.get_world_size()
            dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())
            wandb.log({key: term_list[-1]})

        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean() / dist.get_world_size()
        dist.all_reduce(total_bpd)
        all_bpd.append(total_bpd.item())
        num_complete += dist.get_world_size() * batch.shape[0]
        wandb.log({"bpd": total_bpd})
        wandb.log({"L_intermediate": np.mean(minibatch_metrics["vb"][:,:-1].sum(dim=1).detach().cpu().numpy())})
        wandb.log({"L_0": np.mean(minibatch_metrics["vb"][:,-1].detach().cpu().numpy())})
        wandb.log({"L_T": np.mean(minibatch_metrics["prior_bpd"].detach().cpu().numpy())})
        wandb.log({"mse": np.mean(minibatch_metrics["mse"].detach().cpu().numpy())})
        wandb.log({"xstart_mse": np.mean(minibatch_metrics["xstart_mse"].detach().cpu().numpy())})
        wandb.log({"means_mse": np.mean(minibatch_metrics["means_mse"].detach().cpu().numpy())})

        print("bpd", total_bpd)
        print("L_intermediate", np.mean(minibatch_metrics["vb"][:, :-1].sum(dim=1).detach().cpu().numpy()))
        print("L_0", np.mean(minibatch_metrics["vb"][:, -1].detach().cpu().numpy()))
        print("L_T", np.mean(minibatch_metrics["prior_bpd"].detach().cpu().numpy()))
        print("mse", np.mean(minibatch_metrics["mse"].detach().cpu().numpy()))
        print("xstart_mse", np.mean(minibatch_metrics["xstart_mse"].detach().cpu().numpy()))
        print("means_mse", np.mean(minibatch_metrics["means_mse"].detach().cpu().numpy()))

        print("mse [:-1]", np.mean(minibatch_metrics["mse"][:, :-1].detach().cpu().numpy()))
        print("mse [:200]", np.mean(minibatch_metrics["mse"][:, :200].detach().cpu().numpy()))
        print("mse [200:400]", np.mean(minibatch_metrics["mse"][:, 200:400].detach().cpu().numpy()))
        print("mse [400:600]", np.mean(minibatch_metrics["mse"][:, 400:600].detach().cpu().numpy()))
        print("mse [600:800]", np.mean(minibatch_metrics["mse"][:, 600:800].detach().cpu().numpy()))
        print("mse [800:-1]", np.mean(minibatch_metrics["mse"][:, 800:-1].detach().cpu().numpy()))


        print("means_mse [:-1]", np.mean(minibatch_metrics["means_mse"][:, :-1].detach().cpu().numpy()))
        print("means_mse [:200]", np.mean(minibatch_metrics["means_mse"][:, :200].detach().cpu().numpy()))
        print("means_mse [200:400]", np.mean(minibatch_metrics["means_mse"][:, 200:400].detach().cpu().numpy()))
        print("means_mse [400:600]", np.mean(minibatch_metrics["means_mse"][:, 400:600].detach().cpu().numpy()))
        print("means_mse [600:800]", np.mean(minibatch_metrics["means_mse"][:, 600:800].detach().cpu().numpy()))
        print("means_mse [800:-1]", np.mean(minibatch_metrics["means_mse"][:, 800:-1].detach().cpu().numpy()))




        print("xstart_mse [:-1]", np.mean(minibatch_metrics["xstart_mse"][:, :-1].detach().cpu().numpy()))
        print("means_mse [:-1]", np.mean(minibatch_metrics["means_mse"][:, :-1].detach().cpu().numpy()))

        print(np.mean(minibatch_metrics["vb"][:, :-1].detach().cpu().numpy(), axis=0))

        logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd)}")

    if dist.get_rank() == 0:
        for name, terms in all_metrics.items():
            out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
            logger.log(f"saving {name} terms to {out_path}")
            np.savez(out_path, np.mean(np.stack(terms), axis=0))

    dist.barrier()
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=True, num_samples=1000, batch_size=1, model_path="",
        download_checkpoint="",
        download_step=-1,
        resume_type="ema"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
