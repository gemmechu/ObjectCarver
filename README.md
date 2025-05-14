# ObjectCarver: Semi-automatic segmentation, reconstruction and separation of 3D objects




## [Project page](https://objectcarver.github.io/) | [Paper](https://openreview.net/attachment?id=jHF0Xp9GVu&name=pdf) | [Arxiv](https://arxiv.org/abs/2407.19108) | [Poster](https://objectcarver.github.io/static/ObjectCarver%203DV%20poster.pdf) |[Data (4.2G)](https://drive.google.com/file/d/1xNcUrSACRMY8QS2dJpSbkQsious2-E7q/)
This is the official repo for the implementation of **ObjectCarver: Semi-automatic segmentation, reconstruction and separation of 3D objects**.
<div>
<img src="https://objectcarver.github.io/static/images/method.png" height="500px">
</div>

## Installation
```
git clone https://github.com/gemmechu/ObjectCarver.git
cd ObjectCarver
```
```
conda create -y -n objectcarver python=3.8 && conda activate objectcarver
pip install numpy==1.23.0 scipy trimesh opencv_python scikit-image imageio imageio-ffmpeg pyhocon tqdm icecream configargparse six pymcubes==0.1.2 matplotlib scikit-learn pandas open3d wandb
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tensorboard kornia
conda install -c conda-forge igl
```

## Running ObjectCarver
### 0. Download sample data
```
mkdir data

https://drive.google.com/file/d/1HIS0QWSinuxgTihkpAchpSWZJ9Qlky2d

unzip ./data/scan_3.zip -d ./data/
```

### 1. Get the full Scene
```
. full_train.sh
```

### 2. Mask propagation(for the provided data, you can skip this part since the mask is already generated)

```
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Generate anchor mask by following 
```
training/mask_propagation/generate_anchor.ipynb
```

run the following to generate mask for all the images 
```
python training/mask_propagation/generate_mask.py 
```
### 3. Object Separation
```
. obj_separation_train.sh
```

## Extract surface from trained model
```
. validate_mesh.sh

```


## Acknowledgement
This code depends on the amazing work from [NeuS](https://github.com/Totoro97/NeuS), [SAM](https://github.com/facebookresearch/segment-anything)  and [NeuriS](https://github.com/jiepengwang/NeuRIS). Thanks for these great projects. We would also like to thank Qianyi Wu for his quick email response and for answering our question regarding ObjectSDF++, Kai Zhang and Aditya Chetan for their insightful discussions, and Milky Hassena for helping with the animations.