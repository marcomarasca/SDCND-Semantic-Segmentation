# Semantic Segmentation
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[model_architecture]: ./images/architecture.png "FCN-VGG16 Architecture"
[enc_dec]: ./images/encoder_decoder.png "FCN-VGG16 Architecture"
[scene_understanding]: ./images/scene_segmentation.png "FCN-VGG16 Architecture"
[kitti_dataset]: ./images/kitti_dataset.png "Kitti Dataset"
[aug_dataset]: ./images/augmented_dataset.png "Kitti Dataset Augmented"
[tuning]: ./images/tuning.png "Tuning Parameters"
[tuning_s_off]: ./images/tuning_s_off.png "Scaling OFF"
[tuning_s_on]: ./images/tuning_s_on.png "Scaling ON"
[tuning_aug]: ./images/tuning_aug.png "Scaling ON, Augmented Data"
[tuning_do]: ./images/tuning_do.png "Scaling ON, Augmented Data, Dropout: 0.25"
[tuning_l2reg]: ./images/tuning_l2reg.png "Scaling ON, Augmented Data, L2 Reg: 0.0001"
[tuning_lr]: ./images/tuning_lr.png "Scaling ON, Augmented Data, Learning Rate: 0.0005"
[tuning_lr2]: ./images/tuning_lr2.png "Scaling ON, Augmented Data, Learning Rate: 0.0001"
[tuning_loss_all]: ./images/tensorboard_loss_f.png "Training loss"
[tensorboard_img]: ./images/tensorboard_img.png "Tensorboard Images"
[tensorboard_loss]: ./images/tensorboard_loss.png "Tensorboard Loss"
[tensorboard_iou]: ./images/tensorboard_iou.png "Tensorboard IOU"
[tensorboard_acc]: ./images/tensorboard_acc.png "Tensorboard Accuracy"

[video1_gif]: ./images/video1.gif "Semantic Segmentation"
[video2_gif]: ./images/video2.gif "Semantic Segmentation"

![Gif: Semantic Segmentation][video1_gif]

Overview
---

This repository contains an implementation with [Tensorflow](https://tensorflow.org/) of a Fully Convolutional Network (FCN) used to label image pixels in the context of semantic scene understanding:

![Semantic Scene Segmentation][scene_understanding]

The implementation is based on the [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf) paper by Evan Shelhamer, Jonathan Long and Trevor Darrell (original caffe implementation can be found [here](https://github.com/shelhamer/fcn.berkeleyvision.org)).


![FCN VGG16 Architecture][model_architecture]

The model uses as *encoder* a [VGG16](https://arxiv.org/abs/1409.1556) model, then a *decoder* is added in order to upsample the filters to the final image size, using 1x1 convolutions and transposed convolutions in order to upsample the layers. Additionally skip layers are used to bring in better spatial information from previous layers.

Getting Started
---

This project was implemented using [TensorFlow](https://www.tensorflow.org/) and you'll need a set of dependencies in order to run the code, in particular:

 - [Python 3 (3.6)](https://www.python.org/)
 - [TensorFlow (1.12)](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy 1.1.0](https://www.scipy.org/)
 On pip, use `pip install scipy==1.1.0`
 - [Matplotlib](https://matplotlib.org/)
 - [Imageio](https://imageio.github.io/)
 - [Tqdm](https://pypi.org/project/tqdm/)
 - [Opencv](https://opencv.org/)

Given the complexity of the model a GPU is strongly suggested to train the model; A good and relatively cheap way is to use an EC2 instance on AWS. For example a p2.xlarge instance on EC2 is a good candidate for this type of task (You'll have to ask for an increase in the limits for this type of instance). Alternatively a cheaper instance type (that I used during training) is the GPU graphics instance g3x.xlarge, the instance is relatively slower but the GPU (M60) is newer and faster than the K80 on the p2.xlarge even though it has less memory (8 vs 12).

You can use the official Deep Learning AMI from Amazon that contains most of the required dependencies (See https://docs.aws.amazon.com/dlami/latest/devguide/gs.html) aside from [tqdm](https://pypi.org/project/tqdm/).

The [main.py](./main.py) script can be run as follows:

```bash
$ python model.py [flags]
```

Where flags can be set to:

* **[--data_dir]**: The folder containing the training data (default ./data)
* **[--runs_dir]**: The folder where the output is saved (default ./runs)
* **[--model_folder]**: The folder where the model is saved/loaded (default ./models/[generated_name])
* **[--epochs]**: The number of epochs (default 80)
* **[--batch_size]**: The batch size (default 25)
* **[--dropout]**: The dropout probability (default 0.5)
* **[--learning_rate]**: The learning rate (default 0.0001)
* **[--l2_reg]**: The amount of L2 regularization (defualt 0.001)
* **[--scale]**: True if scaling should be applied to layers 3 and 4 of VGG (default True)
* **[--early_stopping]**: The number of epochs after which the training is stopped if the loss didn't improve (default 4)
* **[--seed]**: Integer used to seed random ops for reproducibility (default None)
* **[--cpu]**: If True disable the GPU (default None)
* **[--tests]**: If True runs the tests (default True)
* **[--train]**: If True runs the training (default True), if a model checkpoint exists in the model_folder the weights will be reloaded
* **[--image]**: Image path to run inference for (default None)
* **[--video]**: Video path to run inference for (defatul None)
* **[--augment]**: Path to the target folder where to save augmented data from data_dir (default None)
* **[--serialize]**: Path of a non existing folder where to save the pb version of the checkpoint saved during training (default None)

### Tensorboard
The script will save summaries for [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) in the *logs* folder:

```bash
$ tensorboard --samples_per_plugin images=0 --logdir=logs
```

The summaries include the training *loss*, *accuracy* and *intersection over union (IOU)* metrics. It will also save images with the predicted result:

![Tensorboard][tensorboard_img]![Tensorboard][tensorboard_loss]

![Tensorboard][tensorboard_iou]![Tensorboard][tensorboard_acc]

### Examples

#### Training

An example to run a training session on 10 epochs with a batch size of 10 and learning rate of 0.001, saving the model into models\my_model: 

```bash
$ python main.py --tests=false --epochs=10 --batch_size=10 --learning_rate=0.001 --model_folder=models\\my_model
```

#### Processing Image

An example of processing a single image image.png using a model saved into models\my_model:

```bash
$ python main.py --tests=false --model_folder=models\\my_model --image=image.png
```

#### Processing Video

An example of processing a video video.mp4 using a model saved into models\my_model:

```bash
$ python main.py --tests=false --model_folder=models\\my_model --video=video.mp4
```

![Video][video2_gif]

#### Dataset aumentation

An example of augmenting the dataset in the data folder ans saving the result in data\augmented (expects the training to be in data\data_road\training):

```bash
$ python main.py --tests=false --data_dir=data --augment=data\\augmented
```

#### Serializing model

An example of serializing a model to a proto buffer in model\my_model\serialized from a checkpoint in models\my_model:

```bash
$ python main.py --tests=false --model_folder=models\\my_model --serialize=models\\my_model\\serialized
```

Dataset
---

In order to train the network we used the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php), that can be downloaded from [here](http://www.cvlibs.net/download.php?file=data_road.zip). It contains both training and testing images, with the ground truth images for the training dataset that are labelled with the correct pixel categorization (road vs non-road):

![Kitti Dataset][kitti_dataset]

Augmentation
---

The Kitti dataset contains 289 labelled samples, in order to improve the model performance it can be easily augmented, the repository contains a python script that simply *mirrors* the images and applies a random amount of *brightness*:

![Kitti Augmented Dataset][aug_dataset]

Training and Testing
---

The training was performed with various hyperparameters, starting from the following *baseline*:

* **Epochs**: 50
* **Batch Size**: 15
* **Learning Rate**: 0.001
* **Dropout**: 0.5
* **L2 Regularization**: 0.001
* **Scaling**: False

Note that **scaling** is a teqnique depicted in the original implementation when they perform what they name "at-once" training, the pooling layers 3 and 4 from the [VGG16](https://arxiv.org/abs/1409.1556) model are scaled before the 1x1 convolution is applied (See https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn8s-atonce/net.py#L65).

![Baseline][tuning_s_off]<br>*Baseline*

Various experiments with different configurations were needed in order to tune the model:

![Training Loss][tuning_loss_all]

And in the following a sample of images with the various configurations:

![Hyperparameters Tuning][tuning]

As we can see scaling smoothen the result better and augmenting the dataset helped in producing more accurate results:

![Baseline][tuning_aug]<br>*Augmented Dataset, Scaling ON*

Using the base learning rate (without decay) the model would converge but stop learning after around 30-40 epochs. When lowering the learning rate on the augmented dataset we could train on **80** epochs which retained the best accuracy:

![Baseline][tuning_lr2]<br>*Augmented Dataset, Scaling ON and Learning Rate 0.0001*

The parameters used for the final training (in one shot):

* **Epochs**: 80
* **Batch Size**: 25
* **Learning Rate**: 0.0001
* **Dropout**: 0.5
* **L2 Regularization**: 0.001
* **Scaling**: True
