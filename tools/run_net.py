# Code adapted from:
# https://github.com/facebookresearch/SlowFast

from lib.utils.misc import launch_job
from lib.utils.parser import load_config, parse_args
from lib.config.defaults import assert_and_infer_cfg

from test_net import test
from train_net import train
from visualization import visualize

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

if __name__ == "__main__":
    main()
