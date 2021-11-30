# Installation

## Requirements
- Python >= 3.8
- Numpy
- PyTorch >= 1.8
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- simplejson: `pip install simplejson`
- GCC >= 4.9
- PyAV: `conda install av -c conda-forge`
- ffmpeg (4.0 is prefereed, will be installed along with PyAV)
- PyYaml: (will be installed along with fvcore)
- tqdm: (will be installed along with fvcore)
- iopath: `pip install -U iopath` or `conda install -c iopath iopath`
- psutil: `pip install psutil`
- OpenCV: `pip install opencv-python`
- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- tensorboard: `pip install tensorboard`

# Data Preparation

#### Dataset and annotations
- Download the M-MiT dataset and annotations from [dataset provider](http://moments.csail.mit.edu/#download).
- Rescale the video to the short edge size of 256 via [preprocess/rescale_video.py](preprocess/rescale_video.py).

```
python rescale_video.py src_video_path dst_video_path
```

- Rename the original annotation (.txt) files for training, validation, and testing set as `train.csv`, `val.csv`, `test.csv`.
- Put all csv files in the same folder, and set `DATA.PATH_TO_DATA_DIR` to the path.
- "Mini M-MiT" training set (93206 videos): we randomly select 300 videos for categories with more than 300 videos and choose all the videos of the remaining categories. Download [train-v1-mini.csv](https://drive.google.com/file/d/1fdltFEgpxiSRJ8WohsxcnSKmAGqRR40I/view?usp=sharing).

#### Generate audio lexicon
- Extract .wav audio files via [preprocess/extract_audio_wav.py](preprocess/extract_audio_wav.py).

```
python extract_audio_wav.py src_video_path dst_audio_path
```

- Install VGGish dependency packages by following [preprocess/vggish/README.md](preprocess/vggish/README.md), and then run following script for generating audio lexicon.

```
bash gen_audio_lexicon.sh
```

This script firstly invokes [preprocess/vggish/mmit_vggish_features.py](preprocess/vggish/mmit_vggish_features.py) to produce the VGGish embeddings, and then generate the audio lexicon via [preprocess/gen_audio_lexicon.sh](preprocess/gen_audio_lexicon.sh)

- Download audio lexicon for "Mini M-MiT": [audio-lexicon-v1-mini.pkl](https://drive.google.com/file/d/12Y0q9bUd1Z5YNZ7I3MdwpX9aawiYDD3R/view?usp=sharing).

#### Generate text lexicon
- Generate the text lexicon via "moments_categories.txt" file (from annotaion files), and run following script:

```
python gen_text_lexicon.py
```

- Download text lexicon for "Mini M-MiT": [text-lexicon-v1-mini.pkl](https://drive.google.com/file/d/12ueMys_2umDw3GhBfu5qp0H-p4zqIUWg/view?usp=sharing).

#### Generate adjacent matrix for GCN
- generate the adjacent matrix for GCN relation learner via "moments_categories.txt" and "trainingSet.txt" (from annotation files), and run following script:

```
python gen_graph_adj.py
```

- Download adjacent matrix for "Mini M-MiT": [graph-adj-v1-mini.pkl](https://drive.google.com/file/d/17ZvNm3vPnRpKIDfVBgv2B3gJjsyert_Y/view?usp=sharing).

#### Structure of data directories

- Set `DATA.PATH_TO_DATA_DIR` as the root folder. Put the annotation files to the root folder, and the lexicon and GCN files to mmit-m3a.

- Name the data files in the form of "name-version.ext". The versions can be v1-mini, v1, and v2. We take Mini M-MiT for an example as following shows:

```
DATA.PATH_TO_DATA_DIR/
  train-v1-mini.csv
  val-v1-mini.csv
  test-v1-mini.csv
  mmit-m3a/
    graph-adj-v1-mini.pkl
    audio-lexicon-v1-mini.pkl
    text-lexicon-v1-mini.pkl
```

