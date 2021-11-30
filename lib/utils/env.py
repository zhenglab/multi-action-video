# Code adapted from:
# https://github.com/facebookresearch/SlowFast

import lib.utils.logging as logging

_ENV_SETUP_DONE = False


def setup_environment():
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True
