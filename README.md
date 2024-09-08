# Towards Dynamic Fine-Refinement Region Proposal Network for Small Object Detection

## :loudspeaker: Introduction
This is the official implementation of our paper titled "Towards Dynamic Fine-Refinement Region Proposal Network for Small Object Detection".

## :ferris_wheel: Dependencies
 - CUDA 11.3
 - Python 3.8
 - PyTorch 1.10.0
 - TorchVision 0.11.0

## :open_file_folder: Datasets
 - SODA-D: [OneDrvie](https://nwpueducn-my.sharepoint.com/:f:/g/personal/gcheng_nwpu_edu_cn/EhXUvvPZLRRLnmo0QRmd4YUBvDLGMixS11_Sr6trwJtTrQ?e=PellK6)
 - Visdrone2019: [website](https://github.com/VisDrone/VisDrone-Datase)

<!-- 
Moreover, this repository is build on MMDetection, please refer to [mmdetection](https://github.com/open-mmlab/mmdetection) for the preparation of corresponding environment.
-->

## 🛠️ Install
This repository is build on MMDetection 2.26.0 which can be installed by running the following scripts. Please ensure that all dependencies have been satisfied before setting up the environment.
```
git clone https://github.com/chnu-cpl/DFRPN.git
cd DFRPN
pip install -v -e .
```

## 🚀 Training
 - Single GPU:
```
python ./tools/train.py ${CONFIG_FILE} 
```

 - Multiple GPUs:
```
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```

## 📈 Evaluation
 - Single GPU:
```
python ./tools/test.py ${CONFIG_FILE} ${WORK_DIR} --eval bbox
```

 - Multiple GPUs:
```
bash ./tools/dist_test.sh ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --eval bbox
```



