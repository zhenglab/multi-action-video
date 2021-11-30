# Get Started

## Running

- To train and test the "Multi-Modal Multi-Action Video Recognition" framework. You can build variant models via setting M3A_MODE, M3A_MODAL_TYPE, M3A_MODAL_JOINT_TYPE.

  ```
  python tools/run_net.py \
    --cfg configs/Mmit/R2PLUS1D_8x2.yaml \
    DATA.PATH_TO_DATA_DIR ${DATA_PATH} \
    DATA.MMIT_VERSION ${MMIT_VERSION} \
    TRAIN.CHECKPOINT_FILE_PATH ${PRETRAIN_PATH} \
    M3A.MODE ${M3A_MODE} \
    M3A.MODAL_TYPE ${M3A_MODAL_TYPE} \
    M3A.MODAL_JOINT_TYPE ${M3A_MODAL_JOINT_TYPE}
  ```
  You can also set the variables (DATA_PATH, MMIT_VERSION, PRETRAIN_PATH) in [scripts/run_mmit_r2plus1d.sh](scripts/run_mmit_r2plus1d.sh), and then run the script.

  ```
  bash scripts/run_mmit_r2plus1d.sh
  ```

- Download the R(2+1)D-34 IG-65M+Kinetics-400 pretrained model: [r2plus1d-34-pretrained-model.pth](https://drive.google.com/file/d/1chCDOz73L-X5fAigcXNGvP8WUNFenB8e/view?usp=sharing). This pretrained model is converted from [facebookresearch/VMZ](https://github.com/facebookresearch/VMZ).

