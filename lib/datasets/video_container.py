# Code adapted from:
# https://github.com/facebookresearch/SlowFast

import av
import pickle


def get_video_container(path_to_vid, multi_thread_decode=False, backend="pyav"):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    if backend == "torchvision":
        with open(path_to_vid, "rb") as fp:
            container = fp.read()
        return container
    elif backend == "pyav":
        container = av.open(path_to_vid)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        return container
    else:
        raise NotImplementedError("Unknown backend {}".format(backend))

def get_sensor_container(path_to_sensor):
    """
    Given the path to the sensor file, return the sensor container.
    Args:
        path_to_sensor_file (str): path to the sensor file.
    Returns:
        container (container): sensor container.
    """
    
    with open(path_to_sensor, 'rb') as f:
        container = pickle.load(f)
    return container
