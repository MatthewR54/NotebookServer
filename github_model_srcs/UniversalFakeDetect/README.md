# TrueMedia.org UniversalFakeDetect Implementation

This repository contains the updated codebase for the "UniversalFakeDetect" model initially introduced by Ojha et al. in "Towards Universal Fake Image Detectors that Generalize Across Generative Models" [[Project Page](https://utkarshojha.github.io/universal-fake-detection/)] [[Paper](https://arxiv.org/abs/2302.10174)]. 

The main contributions of this repo when compared to the original repository are:
- simplified codebase that reduces functionality to TrueMedia.org specific goals
- including additional model architecture support
- migrating to current standards of using HuggingFace CLIP models rather than the clip package directly
- streamlined scripting to train and test models on TrueMedia.org data formats
- support for Docker images and Octo deployment

## Setup

Use the ```requirements.txt``` file to set up an environment:

```
conda create -n ufd-env python=3.10
conda activate ufd-env
which pip # (ensure that the pip path is under the ufd-env environment)
pip install -r requirements.txt
```

## Repository Structure

Rundown of important files and directories:

```models/```: contains the class definitions for trainable networks placed on top of the fixed CLIP backbones.
- ```UniversalFakeDetectv2.py```: the two-layer network with a ReLU activation layer we use for TrueMedia.org models. See [Model Architecture](#model-architecture) for more details.
  
```pretrained_weights/```: contains pretrained weights to either continue training from or to evaluate. Users must place relevant pretrained weights into this directory; they are not automatically populated.

```utils files```: 
- ```argument_utils.py```: handles all parser arguments that are sent when training and/or evaluating models. See file for more descriptions.
- ```model_utils.py```: handles loading model architectures and checkpoints, as well as applying the model over datasets.
- ```data/data_utils.py```: utils to create dataloaders and handle data augmentation.
- ```eval_utils.py```: utils to handle evaluating models, including running inference and gathering evaluation metrics.

```train.py```: Main training script.

```evaluate.py```: Main model evaluation script.

```custommodel.py, Dockerfile, server.py```: required files for deployment. See [Docker Deployment](#Docker-Deployment) for more details.

## Migration to TrueMedia.org's UniversalFakeDetect from original paper implementation

Most functionality and high level usage of the UniversalFakeDetect code remains the same in this repository compared to Ojha et al.

Here is a table of conversions that loosely describe where to find related information, if it has moved:

|UniversalFakeDetect | UniversalFakeDetectV2|
|---|---|
|```data/datasets.py``` | Dataset definition now in ```data/dataset.py```, and utility functions factored out to ```data/data_utils.py```.|
| ```models/clip/``` | ```models/``` now includes separate classes for each network placed on top of the CLIP backbone. We no longer deal with the CLIP architecture ourselves. |
| ```options/``` | All options have been moved to ```argument_utils.py``` to streamline where we find arguments passed into our scripts. | 

## Training for TrueMedia.org model

### Model Architecture

For TrueMedia.org, we have experimented with the architecture placed on top of the CLIP backbone. We refer to this as the UFD-T model (short for UniversalFakeDetect-Transfer), and in its current form, it uses two linear layers on top of the CLIP backbone with a ReLU activation function applied. 

This model architecture can be found under ```models/UniversalFakeDetectv2.py```. In this repo, the ```models``` directory contains the trainable parameters we place on top of the CLIP backbone.

By default, the CLIP backbone we use (from Ojha et al.) is the ```openai/clip-vit-large-patch14```. To use a different **CLIP** backbone, pass in a different HuggingFace ID to the ```--backbone``` parameter when training or evaluating and update the model parameters as necessary (e.g., input dimension referenced in ```models/UniversalFakeDetectv2.py```, or, alternatively, the loaded model in ```load_model()``` from ```model_utils.py```).

### Training with additional data

As of 6/2024, TrueMedia.org retrains/finetunes the UFD-T model with each additional month's data. Typically, this looks like starting from the checkpoint saved in ```pretrained_weights/continued_ft_from_concat_diffusiondb.pth```, which saves a pretrained UFD-T backbone trained on academic datasets, and finetuning using a class-balanced approach using TrueMedia.org data through the last month. Note that ```pretrained_weights/continued_ft_from_concat_20240930.pth``` is the exception to this; it was trained with approximately 350 more reals as this recipe was found to perform higher on the October test set. See the following section on script usage for an example of this. Note that this is not the only possible approach; we could continue to finetune beyond any specific checkpoint instead.

```continued_ft_from_concat_diffusiondb.pth``` was trained on the following fake and real images:

Fake:
- [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb) (16000)
- [StyleGAN2](https://arxiv.org/abs/1912.04958) (8000) 
- [Stable Diffusion Face Dataset](https://github.com/tobecwb/stable-diffusion-face-dataset) (2400 512x512, 2400 768x768, 2400 1024x1024)

Real:
- [Celeb-A-HQ](https://arxiv.org/abs/1710.10196) (random sample of 23200)
- [FFHQ](https://arxiv.org/abs/1812.04948) (random sample of 3200)
- [COCO](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) (random sample of 4800 from train set)


### Script Usage Example
The following trains a UFD-T model, starting from the checkpoint saved in ```pretrained_weights/continued_ft_from_concat_diffusiondb.pth```. It trains the model using the real and fake images stored at ```/tm-images-20240615/images``` and ```/tm-images-20240616-20240715/images```, which are all user-uploaded, labeled images through 2024-07-15. It currently validates/tests the model on the evaluation data stored at ```/tm-images-_20240716-20240815/images```, which are all user-uploaded, labeled images from 20240716 through 20240815. It does *not* train the CLIP backbone, it trains for 10 epochs, calculates the loss every 100 steps, and balances the training data by class, training with an initial learning rate of 0.01.

```
python train.py 
    --name tmp_experiment
    --ckpt pretrained_weights/continued_ft_from_concat_diffusiondb.pth
    --real_list_paths /tm-images-20240615/images/real /tm-images-20240616-20240715/images/real
    --fake_list_paths /tm-images-20240615/images/fake /tm-images-20240616-20240715/images/fake
    --val_real_list_paths /tm-images-_20240716-20240815/images/real
    --val_fake_list_paths /tm-images-_20240716-20240815/images/fake
    --niter 10
    --loss_freq 100
    --class_bal
    --is_train
    --lr 0.01
```
Users should replace all ```--*_list_paths``` arguments with their appropriate datasets.

The following then does a more thorough analysis by evaluating a trained model, stored at ```--ckpt```. Results will be saved to ```--results_folder=/results/tmp_experiment/```. ```--real_list_paths``` and ```--fake_list_paths``` provide the data that the model will be evaluated on.

```
python evaluate.py 
    --ckpt pretrained_weights/finetuned_from_concat_20240715.pth
    --result_folder=results/tmp_experiment
    --real_list_paths /tm-images-_20240716-20240815/images/real
    --fake_list_paths /tm-images-_20240716-20240815/images/fake
```

**Note:** This script requires that training, validation, and testing data live in separate directories.

The exact definitions and default parameters for the scripts can be found under ```argument_utils.py```.
In particular, the ```real_list_paths```, ```fake_list_paths```, ```val_real_list_paths```, and ```val_fake_list_paths``` arguments take in one or more paths to directories of images, separated by a space.

## Docker Deployment

The provided Dockerfile uses the file ```custommodel.py``` to load the model and run inference. Before creating an image, users should confirm that running ```python custommodel.py``` with test input defined in ```main()``` yields the expected results.

Then, to deploy:

Run
``sudo usermod -a -G docker ubuntu``
to allow ubuntu user to execute Docker commands without needing superuser privileges. 

Run the provided Dockerfile to create an image:

```
export DOCKER_REGISTRY="<dockerhub_username>"
# Build the Docker image for runtime
docker build --no-cache -t "$DOCKER_REGISTRY/ufd:20240930" -f Dockerfile .
```

Push your docker image to docker hub
```
docker login

docker push $DOCKER_REGISTRY/ufd:20240930  # remember to update the tag for your version
```

To pull the docker image, 
```
docker pull $DOCKER_REGISTRY/ufd:20240930
```

To debug and view logs, `docker logs ufd`

Run this Docker image locally on a GPU to test that it can run inference as expected:
```
docker run --gpus=all -d --rm -p 80:8000 --env SERVER_PORT=8000  --name "ufd" "$DOCKER_REGISTRY/ufd:20240930"
```

In a separate terminal, run the following command one or more times
```
curl -X GET http://localhost:80/healthcheck
```
until you see {"healthy":true}.

Then, test that inference can be run as expected on local:
```
curl -X POST http://localhost:80/predict \
    -H "Content-Type: application/json" \
    --data '{"file_path":"https://path.to.someimage.jpg"}'
```

To test deployment to a server, 
```
curl -X POST http://<XX.XXX.XXX.XXX>:80/predict -H "Content-Type: application/json" --data '{"file_path":"https://path.to.someimage.jpg"}'
```
