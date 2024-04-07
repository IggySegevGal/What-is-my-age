<h2 align="center">Technion's Deep Learning (046211) Project
</h2> 
<p align="center">
<img src="./xray_61.png" width="300" height="300" />
  </p>
<h4 align="center">
    Iggy Segev Gal:
  
  <a href="https://www.linkedin.com/in/iggy-s-6a53b619a/"><img src="./LinkedIn_icon.png" width="40" height="40"/></a>
    <a href="https://github.com/iggysegevgal"><img src="./GitHub_logo.png" width="40" height="40"/></a>
</a>

<h4 align="center">
    Eyal Gilron:
  
  <a href="https://www.linkedin.com/in/eyal-gilron-9305ab151/"><img src="./LinkedIn_icon.png" width="40" height="40"/></a>
    <a href="https://github.com/eyalgilron"><img src="./GitHub_logo.png" width="40" height="40"/></a>
</a>

Table Of Contents
--
* [Background](#background)
* [File Tree](#file-tree)
* [Dataset](#dataset)
* [Architectures](#architectures)
* [Results](#results)
  * [Densenet with Imagenet](#Densenet-with-Imagenet)
  * [Densenet with chesXnet](#Densenet-with-chesXnet)
  * [Dino](#Dino)
* [Refrences](#refrences)

## Background
In this project we used the NIH-chest-X-rays dataset for age estimation from chest X-rays, we tested both classification and regression tasks, and two different models - Densenet121 and Dino. With Densenet, we tested two different weight initializations from models trained on Imagenet and chesXnet. We found the best hyper-parameters for each combination. We also tested how much of the data can be reducted and still maintain good results.

## File Tree
|File Name | Purpose |
|----------|---------|
|`Config.py`|This file holds all of the project's configurations, parameters such as: which model to currently to train, different scheduler choices, learning rate etc.|
|`train.py`| The main part of the code, this files loads the dataset, calls the training method and then evaluates the results.|
|`Dataloader.py`| Data loader for the NIH-chest x-ray with age detection|
|`Dino.py`|This file contains the dino model class|

## Dataset
The dataset We used the dataset “NIH Chest X-rays” dataset which has 112,120 frontal view chest X-ray images of 30,805 different patients.
We only took one image per patient and only from patients whose ages are in the range of 20-70.
The data consists of 1024x1024 pixel grayscale images.
Number of examples per label:
|Label Name |# Training Examples|
|-----------|-------------------|
|20-30|3707|
|30-40|4892|
|40-50|6319|
|50-60|7138|
|60-70|4721|

Example of chest X-ray image from the dataset:

<img src="./xray.jpg" width="300" height="300" />


## Architectures
For this classification task we trained a few different models to perform the task, the models we used are:
* Densenet with Imagenet
* Densenet with chesXnet
* Dino

## Results

### Densenet with Imagenet
Regression:
<p float="left">
  <img src=".png" width="300" height="300" />
  <img src=".png" width="300" height="300" /> 
</p>
Classification:
<p float="left">
  <img src=".png" width="300" height="300" />
  <img src=".png" width="300" height="300" /> 
</p>
Final Test Accuracy = 

### Densenet with chesXnet
Regression:
<p float="left">
  <img src=".png" width="300" height="300" />
  <img src=".png" width="300" height="300" /> 
</p>
Classification:
<p float="left">
  <img src=".png" width="300" height="300" />
  <img src=".png" width="300" height="300" /> 
</p>
Final Test Accuracy = 

### Dino
Regression:
<p float="left">
  <img src=".png" width="300" height="300" />
  <img src=".png" width="300" height="300" /> 
</p>
Classification:
<p float="left">
  <img src=".png" width="300" height="300" />
  <img src=".png" width="300" height="300" /> 
</p>
Final Test Accuracy = 

## Refrences
[1] NIH Dataset: https://www.kaggle.com/datasets/nih-chest-xrays

[2] chesXnet Model: https://github.com/arnoweng/CheXNet

[3] Dino Model: https://github.com/facebookresearch/dinov2

[4] Idea from: https://www.nature.com/articles/s43856-022-00220-6

