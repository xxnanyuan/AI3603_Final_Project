# AI3603_Final_Project: Autonomous Driving Simulation Experiment

## 1 Introduction

This is final project for AI3603 in SJTU.

In this project, we run autonomous driving simulation exper-
iment. Our task is implement reinforcement learning to control
the vehicle move as fast as possible while ensuring safety to
accomplish specified tasks such as tracking and parking.

## 2 Environment

We use [conda](https://anaconda.org/anaconda/conda) to make and manage python environments.

1. `conda create -n highway python=3.8`
2. `conda activate highway`
3. `pip install -r requirements.txt`

## 3 Code Structure

``` shell_tree
.
├── AI3603_2023_Project_1__Autonomous_Driving_Simulation_Experiment_Report.pdf
├── README.md
├── ReplayBuffer.py
├── SAC.py
├── WCSAC.py
├── eval_files/
├── firgure/
├── makeVecEnv.py
├── otherFailAttempt
├── requirements.txt
├── runHighway.sh
├── runIntersection.sh
├── runParking.sh
├── runRaceTrack_wcsac.sh
├── runRacetrack.sh
├── train/
├── trainHighway.py
├── trainIntersection.py
├── trainIntersectionTurnRight.py
├── trainParking.py
├── trainRaceTrack_wcsac.py
├── trainRacetrack.py
├── utils.py
└── utils_wcsac.py
```

## 4 Running Guide

### 4.1 Train

To train the model, run:

```shell
bash runIntersection.sh
bash runHighway.sh
bash runParking.sh
bash runRaceTrach.sh
```

You could specify detail hyper-parameters for algorithms and training in shell script. The default value is in `utils.py`.

The training result will store in dir "train/".

### 4.2 Eval

We give an example here. You have train a model and the model is stored in `train/highway-v0/2023-12-22 22-59-11/`. Firstly move the model to `eval_files/models/highway/`.

```shell
cp -r train/highway-v0/2023-12-22 22-59-11/ eval_files/models/highway/
```

Then change the dir to eval_files and run:

```shell
cd eval_files/
python eval_highway.py
```

To show the video during eval, uncomment line 101 in `eval_highway.py`. To record the video, uncomment line 7~9 in `eval_highway.py`.

## 5 Result

Our best model with hyper-parameters is store in `eval_file/`. See the video in [eval_files/videos/](eval_files/videos/).

## 6 Problem and Solution

Here are some problem we meet in the project.

### 6.1 X server error

Problem: 'GLIBCXX_3.4.30' not found for librosa in conda virtual environment

Solution: [STFW](https://bcourses.berkeley.edu/courses/1478831/pages/glibcxx-missing), we know this problem is caused by the version of GLIBCXX. The version of GLIBCXX in conda is 3.4.29, by the system require 3.4.30. So just search the available version in local environment and make a sysbolic link.

``` shell
cd FoldofVirtualEnvironmetLib
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```
