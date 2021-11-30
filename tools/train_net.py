# Code adapted from:
# https://github.com/facebookresearch/SlowFast

import numpy as np
import pprint
import torch
import random
# from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from lib.utils.precise_bn_helper import get_bn_modules, update_bn_stats

import lib.models.losses as losses
import lib.models.optimizer as optim
import lib.utils.checkpoint as cu
import lib.utils.distributed as du
import lib.utils.logging as logging
import lib.utils.metrics as metrics
import lib.utils.misc as misc
from lib.datasets import loader
from lib.models import build_model
from lib.utils.meters import TrainMeter, ValMeter, EpochTimer

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, grad_scaler=None
):
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.ENABLE_AMP):
            preds = model(inputs)
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    
            # Compute the loss.
            loss = loss_fun(preds, labels)
    
            # check Nan Loss.
            misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        if cfg.TRAIN.ENABLE_AMP:            
            # mix precision optimize
            # loss.backward()
            # optimizer.step()

            # mix precision optimize using grad_scaler
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:  
            # Perform the backward pass.
            loss.backward()
            # Update the parameters.
            optimizer.step()

        top1_err, top5_err = None, None
        if cfg.DATA.MULTI_LABEL:
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                [loss] = du.all_reduce([loss])
            loss = loss.item()
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, top1_err, top5_err = du.all_reduce(
                    [loss, top1_err, top5_err]
                )

            # Copy the stats from GPU to CPU (sync point).
            loss, top1_err, top5_err = (
                loss.item(),
                top1_err.item(),
                top5_err.item(),
            )

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.ENABLE_AMP):
            preds = model(inputs)

        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                preds, labels = du.all_gather([preds, labels])
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

            # Combine the errors across the GPUs.
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])

            # Copy the errors from GPU to CPU (sync point).
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )

        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True, use_amp=False):
    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters, use_amp)


def build_trainer(cfg):
    # Build the video model and print model statistics.
    model = build_model(cfg)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    optimizer = optim.construct_optimizer(model, cfg)
    
    grad_scaler = None
    if cfg.TRAIN.ENABLE_AMP:
        grad_scaler = torch.cuda.amp.GradScaler()

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, grad_scaler
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
                cfg.TRAIN.ENABLE_AMP,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
