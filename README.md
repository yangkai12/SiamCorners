# SiamCorners
This is an official implemention for “[SiamCorners: Siamese Corner Networks for VisualTracking](https://arxiv.org/pdf/2104.07303.pdf)”. The code will be made public, be patient. 


![image](https://user-images.githubusercontent.com/25238475/115003176-15eb7400-9e95-11eb-9a51-275235429679.png)
The overview of our SiamCorners architecture, which includes the Siamese feature extractor followed by the top-left corner and bottom-right corner
branches in parallel. 

## Dependencies
* Python 3.7
* PyTorch 1.0.0
* numpy
* CUDA 10
* skimage
* matplotlib
* GCC 4.9.2 or above
## Prepare training dataset
Prepare training dataset, detailed preparations are listed in [training_dataset](training_dataset) directory.
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/) ([BaiduYun](https://pan.baidu.com/s/1nXe6cKMHwk_zhEyIm2Ozpg), extract code: h964.)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [LaSOT](https://cis.temple.edu/lasot/)
* [GOT10k](http://got-10k.aitestunion.com/)
### Compiling Corner Pooling Layers
Compile the C++ implementation of the corner pooling layers. (GCC4.9.2 or above is required.)
```
cd <SiamCorners dir>/pysot/models/corners/py_utils/_cpools
python setup.py install --user
```

### Compiling NMS
Compile the NMS code which are originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx).
```
cd <SiamCorners dir>/pysot/tracker/external
make
```
#### Training:
```bash
CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=2333 \
    ../../tools/train.py --cfg config.yaml
```

#### Testing:
```
python ../tools/test.py 
```

### Citation
If you're using this code in a publication, please cite our paper.

	@InProceedings{SiamCorners,
	author = {Kai Yang, Zhenyu He, Wenjie Pei, Zikun Zhou, Xin Li, Di Yuan and Haijun Zhang},
	title = {SiamCorners: Siamese Corner Networks for Visual Tracking},
	booktitle = {IEEE Transactions on Multimedia},
	month = {April},
	year = {2021}
	}

## Acknowledgment
Our anchor-free tracker is based on [PySot](https://github.com/STVIR/pysot) and [CornerNet](https://github.com/princeton-vl/CornerNet-Lite). We sincerely thank the authors Bo Li and Hei Law for providing these great works.

## Contact
If you have any questions, please feel free to contact yangkaik88@163.com
