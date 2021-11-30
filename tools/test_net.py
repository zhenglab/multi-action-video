# Code adapted from:
# https://github.com/facebookresearch/SlowFast

import numpy as np
import os
import pickle
import torch
import random
from iopath.common.file_io import g_pathmgr

import lib.utils.checkpoint as cu
import lib.utils.distributed as du
import lib.utils.logging as logging
import lib.utils.misc as misc
from lib.datasets import loader
from lib.models import build_model
from lib.utils.meters import TestMeter

logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        # Perform the forward pass.
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.ENABLE_AMP):
            preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx]
            )
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            preds.detach(), labels.detach(), video_idx.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    all_preds = test_meter.video_preds.clone().detach()
    all_labels = test_meter.video_labels
    if cfg.NUM_GPUS:
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu()

    if cfg.TEST.SAVE_RESULTS_PATH != "":
        save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

        if du.is_root_proc():
            with g_pathmgr.open(save_path, "wb") as f:
                pickle.dump([all_preds, all_labels], f)

        logger.info(
            "Successfully saved prediction results to {}".format(save_path)
        )

    test_meter.finalize_metrics()
    return test_meter

def test(cfg):
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
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        test_loader.dataset.num_videos
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        test_loader.dataset.num_videos
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg)
