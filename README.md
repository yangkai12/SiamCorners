# SiamCorners
This is an official implemention for “[SiamCorners](https://arxiv.org/pdf/2104.07303.pdf): Siamese Corner Networks for VisualTracking”. The code will be made public, be patient. 


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

### Compiling Corner Pooling Layers
Compile the C++ implementation of the corner pooling layers. (GCC4.9.2 or above is required.)
```
cd <SiamCorners dir>/core/models/py_utils/_cpools/
python setup.py install --user
```

### Compiling NMS
Compile the NMS code which are originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx).
```
cd <SiamCorners dir>/core/external
make
```

### Citation
If you're using this code in a publication, please cite our paper.

	@InProceedings{SiamCorners,
	author = {Kai Yang, Zhenyu He, Wenjie Pei, Zikun Zhou, Xin Li, Di Yuan and Haijun Zhang},
	title = {SiamCorners: Siamese Corner Networks for VisualTracking},
	booktitle = {IEEE Transactions on Multimedia},
	month = {April},
	year = {2021}
	}

## Acknowledgment
Our transformer-assisted tracker is based on [PySot](https://github.com/STVIR/pysot) and [CornerNet](https://github.com/princeton-vl/CornerNet-Lite). We sincerely thank the authors Bo Li and Hei Law for providing these great work.

## Contact
If you have any questions, please feel free to contact yangkaik88@163.com
