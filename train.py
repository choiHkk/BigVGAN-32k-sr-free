# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


import logging
import warnings

import argparse
import itertools
import json
import os
import time

import auraloss
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchaudio as ta
import wandb
from pesq import pesq
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from bigvgan import BigVGAN
from discriminators import (
    MultiBandDiscriminator,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from env import AttrDict, build_env
from loss import (
    MultiScaleMelSpectrogramLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from meldataset import MAX_WAV_VALUE, Collator, get_dataset, mel_spectrogram
from utils import load_checkpoint, save_checkpoint, scan_checkpoint

warnings.simplefilter(action="ignore", category=FutureWarning)
torch.backends.cudnn.benchmark = False


def train(rank, a, h):
    if h.num_gpus > 1:
        # initialize distributed
        init_process_group(
            backend=h.dist_config["dist_backend"],
            init_method=h.dist_config["dist_url"],
            world_size=h.dist_config["world_size"] * h.num_gpus,
            rank=rank,
        )

    if rank == 0:
        wandb.init(
            project=h.project_name,
            name=h.experiment_name,
            config=h,
            dir=h.checkpoint_path,
        )

    # Set seed and device
    torch.cuda.manual_seed(h.seed)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank:d}")

    # Define BigVGAN generator
    generator = BigVGAN(h).to(device)

    # Define discriminators. MPD is used by default
    mpd = MultiPeriodDiscriminator(h).to(device)

    # Define additional discriminators. BigVGAN-v1 uses UnivNet's MRD as default
    # New in BigVGAN-v2: option to switch to new discriminators: MultiBandDiscriminator / MultiScaleSubbandCQTDiscriminator
    if h.get("use_mbd_instead_of_mrd", False):  # Switch to MBD
        print(
            "[INFO] using MultiBandDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
        )
        # Variable name is kept as "mrd" for backward compatibility & minimal code change
        mrd = MultiBandDiscriminator(h).to(device)
    elif h.get("use_cqtd_instead_of_mrd", False):  # Switch to CQTD
        print(
            "[INFO] using MultiScaleSubbandCQTDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
        )
        mrd = MultiScaleSubbandCQTDiscriminator(h).to(device)
    else:  # Fallback to original MRD in BigVGAN-v1
        mrd = MultiResolutionDiscriminator(h).to(device)

    # New in BigVGAN-v2: option to switch to multi-scale L1 mel loss
    if h.get("use_multiscale_melloss", False):
        print(
            "[INFO] using multi-scale Mel l1 loss of BigVGAN-v2 instead of the original single-scale loss"
        )
        fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
            sampling_rate=h.sampling_rate
        )  # NOTE: accepts waveform as input
    else:
        fn_mel_loss_singlescale = F.l1_loss

    # Print the model & number of parameters, and create or scan the latest checkpoint from checkpoints directory
    if rank == 0:
        print(generator)
        print(mpd)
        print(mrd)
        print(f"Generator params: {sum(p.numel() for p in generator.parameters())}")
        print(f"Discriminator mpd params: {sum(p.numel() for p in mpd.parameters())}")
        print(f"Discriminator mrd params: {sum(p.numel() for p in mrd.parameters())}")
        os.makedirs(h.checkpoint_path, exist_ok=True)
        print(f"Checkpoints directory: {h.checkpoint_path}")

    if os.path.isdir(h.checkpoint_path):
        # New in v2.1: If the step prefix pattern-based checkpoints are not found, also check for renamed files in Hugging Face Hub to resume training
        cp_g = scan_checkpoint(
            h.checkpoint_path, prefix="g_", renamed_file="bigvgan_generator.pt"
        )
        cp_do = scan_checkpoint(
            h.checkpoint_path,
            prefix="do_",
            renamed_file="bigvgan_discriminator_optimizer.pt",
        )

    # Load the latest checkpoint if exists
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, "cpu")
        state_dict_do = load_checkpoint(cp_do, "cpu")
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        mrd.load_state_dict(state_dict_do["mrd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    # Initialize DDP, optimizers, and schedulers
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    # Define training and validation datasets
    dataset = get_dataset(h.data_patterns, rank=rank, cache_dir=h.cache_dir)
    dataset = dataset.train_test_split(test_size=h.test_size)
    trainset = dataset["train"]

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=Collator(h, split=True),
    )

    if rank == 0:
        validset = dataset["test"]
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
            collate_fn=Collator(h, split=False),
        )

    """
    Validation loop, "mode" parameter is automatically defined as seen_(name of the dataset).
    If the name of the dataset contains "nonspeech", it skips PESQ calculation to prevent errors 
    """

    def validate(rank, a, h, loader, mode=""):
        assert rank == 0, "validate should only run on rank=0"
        generator.eval()
        torch.cuda.empty_cache()

        val_err_tot = 0
        val_pesq_tot = 0
        val_mrstft_tot = 0

        # Modules for evaluation metrics
        pesq_resampler = ta.transforms.Resample(h.sampling_rate, 16000).cuda()
        loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device="cuda")

        with torch.no_grad():
            print(f"step {steps} speaker validation...")

            # Loop over validation set and compute metrics
            for j, batch in enumerate(tqdm(loader)):
                x, reference_x, y, y_mel = batch
                y = y.to(device)
                target_size = y_mel.shape[-1]

                if h.get("use_reference_encoder", False):
                    reference_x = reference_x.to(device)

                if hasattr(generator, "module"):
                    y_g_hat = generator.module(
                        x.to(device),
                        target_size=target_size,
                        reference_x=reference_x,
                    )
                else:
                    y_g_hat = generator(
                        x.to(device),
                        target_size=target_size,
                        reference_x=reference_x,
                    )

                y_mel = y_mel.to(device, non_blocking=True)
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1),
                    h.n_fft,
                    h.num_mels,
                    h.sampling_rate,
                    h.hop_size,
                    h.win_size,
                    h.fmin,
                    h.fmax_loss,
                )
                min_t = min(y_mel.size(-1), y_g_hat_mel.size(-1))
                val_err_tot += F.l1_loss(
                    y_mel[..., :min_t], y_g_hat_mel[..., :min_t]
                ).item()

                # PESQ calculation. only evaluate PESQ if it's speech signal (nonspeech PESQ will error out)
                # Resample to 16000 for pesq
                y_16k = pesq_resampler(y)
                y_g_hat_16k = pesq_resampler(y_g_hat)
                y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
                y_g_hat_int_16k = (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()

                try:
                    val_pesq_tot += pesq(
                        16000, y_int_16k, y_g_hat_int_16k.squeeze(0), "wb"
                    )
                except Exception as e:
                    logging.error(f"PESQ calculation failed: {e}")

                # MRSTFT calculation
                min_t = min(y.size(-1), y_g_hat.size(-1))
                val_mrstft_tot += loss_mrstft(
                    y_g_hat[..., :min_t], y[..., :min_t]
                ).item()

                # Log audio and figures to Tensorboard
                if j % a.eval_subsample == 0:  # Subsample every nth from validation set
                    wandb_log_dict = {}
                    if steps >= 0:
                        wandb_log_dict[f"gt_{mode}_audio/y_{j}"] = wandb.Audio(
                            y[0].view(-1).detach().cpu().numpy(),
                            sample_rate=h.sampling_rate,
                        )

                    wandb_log_dict[f"generated_{mode}_audio/y_hat_{j}"] = wandb.Audio(
                        y_g_hat[0].view(-1).detach().cpu().numpy(),
                        sample_rate=h.sampling_rate,
                    )

                    # Spectrogram of synthesized audio
                    mel_spectrogram(
                        y_g_hat.squeeze(1),
                        h.n_fft,
                        h.num_mels,
                        h.sampling_rate,
                        h.hop_size,
                        h.win_size,
                        h.fmin,
                        h.fmax,
                    )
                    wandb.log(wandb_log_dict, step=steps)

            val_err = val_err_tot / (j + 1)
            val_pesq = val_pesq_tot / (j + 1)
            val_mrstft = val_mrstft_tot / (j + 1)
            # Log evaluation metrics to Tensorboard

            scalar_dict = {
                f"validation_{mode}/mel_spec_error": val_err,
                f"validation_{mode}/pesq": val_pesq,
                f"validation_{mode}/mrstft": val_mrstft,
            }
            wandb.log(scalar_dict, step=steps)

        generator.train()

    # If the checkpoint is loaded, start with validation loop
    if steps != 0 and rank == 0 and not a.debug:
        validate(rank, a, h, validation_loader, mode="train_loop")

    # Exit the script if --evaluate is set to True
    if a.evaluate:
        exit()

    # Main training loop
    generator.train()
    mpd.train()
    mrd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print(f"Epoch: {epoch + 1}")

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, reference_x, y, y_mel = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            target_size = y_mel.shape[-1]
            if not h.get("use_multiscale_melloss", False):
                y_mel = y_mel.to(device, non_blocking=True)

            if h.get("use_reference_encoder", False):
                reference_x = reference_x.to(device, non_blocking=True)

            y = y.unsqueeze(1)

            y_g_hat = generator(x, target_size=target_size, reference_x=reference_x)
            if not h.get("use_multiscale_melloss", False):
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1),
                    h.n_fft,
                    h.num_mels,
                    h.sampling_rate,
                    h.hop_size,
                    h.win_size,
                    h.fmin,
                    h.fmax_loss,
                )

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )

            # MRD
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )

            loss_disc_all = loss_disc_s + loss_disc_f

            # Set clip_grad_norm value
            clip_grad_norm = h.get("clip_grad_norm", 1000.0)  # Default to 1000

            # Whether to freeze D for initial training steps
            loss_disc_all.backward()
            grad_norm_mpd = torch.nn.utils.clip_grad_norm_(
                mpd.parameters(), clip_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(mrd.parameters(), clip_grad_norm)
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            lambda_melloss = h.get(
                "lambda_melloss", 45.0
            )  # Defaults to 45 in BigVGAN-v1 if not set
            if h.get("use_multiscale_melloss", False):  # uses wav <y, y_g_hat> for loss
                loss_mel = fn_mel_loss_multiscale(y, y_g_hat) * lambda_melloss
            else:  # Uses mel <y_mel, y_g_hat_mel> for loss
                loss_mel = fn_mel_loss_singlescale(y_mel, y_g_hat_mel) * lambda_melloss

            # MPD loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

            # MRD loss
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(y, y_g_hat)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                generator.parameters(), clip_grad_norm
            )
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    mel_error = (
                        loss_mel.item() / lambda_melloss
                    )  # Log training mel regression loss to stdout
                    print(
                        f"Steps: {steps:d}, "
                        f"Gen Loss Total: {loss_gen_all:4.3f}, "
                        f"Mel Error: {mel_error:4.3f}, "
                        f"s/b: {time.time() - start_b:4.3f} "
                        f"lr: {optim_g.param_groups[0]['lr']:4.7f} "
                        f"grad_norm_g: {grad_norm_g:4.3f}"
                    )

                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = f"{h.checkpoint_path}/g_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "generator": (
                                generator.module if h.num_gpus > 1 else generator
                            ).state_dict()
                        },
                    )
                    checkpoint_path = f"{h.checkpoint_path}/do_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                            "mrd": (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "optim_d": optim_d.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    mel_error = (
                        loss_mel.item() / lambda_melloss
                    )  # Log training mel regression loss to tensorboard

                    scalar_dict = {
                        "training/gen_loss_total": loss_gen_all.item(),
                        "training/mel_spec_error": mel_error,
                        "training/fm_loss_mpd": loss_fm_f.item(),
                        "training/gen_loss_mpd": loss_gen_f.item(),
                        "training/disc_loss_mpd": loss_disc_f.item(),
                        "training/grad_norm_mpd": grad_norm_mpd,
                        "training/fm_loss_mrd": loss_fm_s.item(),
                        "training/gen_loss_mrd": loss_gen_s.item(),
                        "training/disc_loss_mrd": loss_disc_s.item(),
                        "training/grad_norm_g": grad_norm_g,
                        "training/learning_rate_d": scheduler_d.get_last_lr()[0],
                        "training/learning_rate_g": scheduler_g.get_last_lr()[0],
                        "training/epoch": epoch + 1,
                    }
                    wandb.log(scalar_dict, step=steps)

                # Validation
                if steps % a.validation_interval == 0:
                    # Plot training input x so far used
                    if not a.debug:
                        validate(rank, a, h, validation_loader, mode="train_loop")

            steps += 1

            # BigVGAN-v2 learning rate scheduler is changed from epoch-level to step-level
            scheduler_g.step()
            scheduler_d.step()

        if rank == 0:
            print(
                f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n"
            )


def main():
    print("Initializing Training Process..")

    parser = argparse.ArgumentParser()

    parser.add_argument("--group_name", default=None)
    parser.add_argument(
        "--config",
        default="./configs/bigvgan_reference.json",
    )

    parser.add_argument("--training_epochs", default=100000, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=1000, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    parser.add_argument("--validation_interval", default=1000, type=int)

    parser.add_argument(
        "--debug",
        default=False,
        type=bool,
        help="debug mode. skips validation loop throughout training",
    )
    parser.add_argument(
        "--evaluate",
        default=False,
        type=bool,
        help="only run evaluation from checkpoint and exit",
    )
    parser.add_argument(
        "--eval_subsample",
        default=50,
        type=int,
        help="subsampling during evaluation loop",
    )

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    build_env(a.config, "config.json", h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print(f"Batch size per GPU: {h.batch_size}")
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(
            train,
            nprocs=h.num_gpus,
            args=(
                a,
                h,
            ),
        )
    else:
        train(0, a, h)


if __name__ == "__main__":
    main()
