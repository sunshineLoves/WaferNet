import wandb
import logging
import os
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from timm import create_model
from data import create_dataset, create_dataloader
from models import DMemSeg
from loss import FocalLoss, EntropyLoss
from train import training
from log import setup_default_logging
from utils import torch_seed
from scheduler import CosineAnnealingWarmupRestarts

os.environ['WANDB_MODE'] = 'offline'

logger = logging.getLogger('train')


def run(config):
    # setting seed and device
    setup_default_logging()
    torch_seed(config.SEED)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info('Device: {}'.format(device))

    # save_dir
    config.EXP_NAME = config.EXP_NAME + f"-{config.DATASET.target}"
    save_dir = os.path.join(config.RESULT.save_dir, config.EXP_NAME)
    os.makedirs(save_dir, exist_ok=True)

    # wandb
    if config.TRAIN.use_wandb:
        wandb.init(name=config.EXP_NAME, project='MemSeg', config=OmegaConf.to_container(config))

    # build datasets
    train_set = create_dataset(
        datadir=config.DATASET.datadir,
        target=config.DATASET.target,
        is_train=True,
        resize=config.DATASET.resize,
        texture_source_dir=config.DATASET.texture_source_dir,
        structure_grid_size=config.DATASET.structure_grid_size,
        transparency_range=config.DATASET.transparency_range,
        perlin_scale=config.DATASET.perlin_scale,
        min_perlin_scale=config.DATASET.min_perlin_scale,
        perlin_noise_threshold=config.DATASET.perlin_noise_threshold,
        use_mask=config.DATASET.use_mask,
        bg_threshold=config.DATASET.bg_threshold,
        bg_reverse=config.DATASET.bg_reverse
    )

    test_set = create_dataset(
        datadir=config.DATASET.datadir,
        target=config.DATASET.target,
        is_train=False,
        resize=config.DATASET.resize
    )

    # build dataloader
    train_dataloader = create_dataloader(
        dataset=train_set,
        train=True,
        batch_size=config.DATALOADER.batch_size,
        num_workers=config.DATALOADER.num_workers
    )

    test_dataloader = create_dataloader(
        dataset=test_set,
        train=False,
        batch_size=config.DATALOADER.batch_size,
        num_workers=config.DATALOADER.num_workers
    )

    # build feature extractor
    feature_extractor = create_model(
        config.MODEL.feature_extractor_name,
        pretrained=False,
        features_only=True
    ).to(device)
    feature_extractor.load_state_dict(torch.load(f'pretrained/{config.MODEL.feature_extractor_name}.pth'))
    # freeze weight of layer 1,2,3
    for layer in ['layer1', 'layer2', 'layer3']:
        for param in feature_extractor[layer].parameters():
            param.requires_grad = False

    # build DMemSeg
    model = DMemSeg(
        feature_extractor=feature_extractor,
        num_memory=config.MEMORY.memory_size,
        feature_shapes=[
            (64, 64, 64),
            (128, 32, 32),
            (256, 16, 16)
        ]
    ).to(device)

    # Set training
    l1_criterion = nn.L1Loss()
    f_criterion = FocalLoss(
        gamma=config.TRAIN.focal_gamma,
        alpha=config.TRAIN.focal_alpha
    )
    entropy_criterion = EntropyLoss(epsilon=config.TRAIN.epsilon)

    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.OPTIMIZER.lr,
        weight_decay=config.OPTIMIZER.weight_decay
    )

    if config['SCHEDULER']['use_scheduler']:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=config.TRAIN.num_training_steps,
            max_lr=config.OPTIMIZER.lr,
            min_lr=config.SCHEDULER.min_lr,
            warmup_steps=int(config.TRAIN.num_training_steps * config.SCHEDULER.warmup_ratio)
        )
    else:
        scheduler = None

    # Fitting model
    training(
        model=model,
        num_training_steps=config.TRAIN.num_training_steps,
        train_dataloader=train_dataloader,
        valid_dataloader=test_dataloader,
        criterion=(l1_criterion, f_criterion, entropy_criterion),
        loss_weights=(config.TRAIN.l1_weight, config.TRAIN.focal_weight, config.TRAIN.entropy_weight),
        optimizer=optimizer,
        scheduler=scheduler,
        log_interval=config.LOG.log_interval,
        eval_interval=config.LOG.eval_interval,
        save_dir=save_dir,
        device=device,
        use_wandb=config.TRAIN.use_wandb
    )


if __name__ == '__main__':
    args = OmegaConf.from_cli()
    # load default config
    configs = OmegaConf.load(args.configs)
    del args['configs']

    # merge config with new keys
    configs = OmegaConf.merge(configs, args)

    # target config
    target_config = OmegaConf.load(configs.DATASET.anomaly_mask_info)
    configs.DATASET = OmegaConf.merge(configs.DATASET, target_config[configs.DATASET.target])

    print(OmegaConf.to_yaml(configs))

    run(configs)
