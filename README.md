# PixHtLab 
This is the source code([[**Paper**](https://arxiv.org/pdf/2303.00137.pdf)]  [[**Demo**](#inference)]  [[**Training**](#training)]) repository for three papers:  


* [(CVPR 2023 highlight) PixHt-Lab: Pixel Height Based Light Effect Generation for Image Compositing](https://arxiv.org/pdf/2303.00137.pdf)

<p align="center">
  <img src="Figs/more_results2.png" />
</p>
 
* [(ECCV 2022) Controllable Shadow Generation Using Pixel Height Maps](https://arxiv.org/pdf/2207.05385.pdf)

<p align="center">
  <img src="Figs/SSG.gif" />
</p>

 
* [(CVPR 2021 oral) SSN: Soft Shadow Network for Image Composition](https://arxiv.org/abs/2007.08211)

<p align="center">
  <img src="Figs/ssn_teaser.png" />
</p>

# SSN  
We released the training codes, a gradio demo (also hosted in huggingface) and the training dataset. 

## Dataset 
The training dataset can created by code provided in the SSN [old repo](https://github.com/ShengCN/SSN_SoftShadowNet).

We provide a precomputed dataset composed of 558 different 3D models, which can be downloaded from our huggingface dataset repo. 
See the [**ssn_dataset.hdf5**](https://huggingface.co/datasets/ysheng/SSN-SSG-PixHtLab/resolve/main/SSN/Dataset) file.


## Demo 
A gradio-based GUI demo is provided in the `Demo/SSN` folder. First, download the [weight](https://huggingface.co/datasets/ysheng/SSN-SSG-PixHtLab/tree/main/SSN/weights) it used.
Then run the following code to see the demo: 
``` bash
python app.py
```

## Training 
The training codes are under `Train` folder. To reproduce the training process, first prepare the dataset discussed above, then run the following command: 

``` bash 
cd Train 

# before running, try to check the setting in SSN.yaml.
python app/Trainer.py --config configs/SSN.yaml
```


# Updates
- [x] [2023-03-16] Basic setup. 
- [x] SSN dataset/demo/inference/training 
- [ ] Python environment setup + a docker image 
- [ ] SSG/PixhtLab dataset/demo/inference/training 

# License
This code repo can only be used for non-commercial use only. 

# Citation
If you think the code/dataset is useful, please remember to cite the three papers: 
```
@inproceedings{sheng2023pixht,
  title={PixHt-Lab: Pixel Height Based Light Effect Generation for Image Compositing},
  author={Sheng, Yichen and Zhang, Jianming and Philip, Julien and Hold-Geoffroy, Yannick and Sun, Xin and Zhang, He and Ling, Lu and Benes, Bedrich},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16643--16653},
  year={2023}
}

@inproceedings{sheng2022controllable,
  title={Controllable shadow generation using pixel height maps},
  author={Sheng, Yichen and Liu, Yifan and Zhang, Jianming and Yin, Wei and Oztireli, A Cengiz and Zhang, He and Lin, Zhe and Shechtman, Eli and Benes, Bedrich},
  booktitle={European Conference on Computer Vision},
  pages={240--256},
  year={2022},
  organization={Springer}
}

@inproceedings{sheng2021ssn,
  title={SSN: Soft shadow network for image compositing},
  author={Sheng, Yichen and Zhang, Jianming and Benes, Bedrich},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4380--4390},
  year={2021}
}
```


