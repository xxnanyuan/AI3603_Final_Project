# AI3603_Project

Group Members: [Gonghu Shang](https://github.com/xxnanyuan)

## 1 Introduction

This is final project for AI3603 in SJTU.

In the [highwayEnv](https://github.com/Farama-Foundation/HighwayEnv) simulation environment, utilizing reinforcement learning and similar methods, drive the vehicle as swiftly as possible while ensuring safety to accomplish specified tasks such as tracking and parking. Our objective within highway, intersection, and race track scenarios is to move as quickly as possible while maintaining safety. The task in parking scenarios involves parking the vehicle at specific locations.

## 2 Environment

We use [conda](https://anaconda.org/anaconda/conda) to make and manage python environments.

1. `conda create -n highway python=3.8`
2. `conda activate highway`
3. `pip install -r requirements.txt`

## 3 Structure

``` shell_tree
.
├── README.md
├── ReplayBuffer.py
├── SAC.py
├── eval/
│   ├── eval_intersection.py
│   ├── eval_parking.py
│   └── eval_racetrack.py
├── myeval/ # dir to store eval log file
├── train/  # dir to store train log file
├── evalApi.py
├── makeEnv.py
├── requirements.txt
├── train.py
├── trainParking.py
└── utils.py
```

## 4 Running Guide

### 4.1 Train

There are many difference between parking env and others, so I spilt them to two file.

You can specify args in utils.

To train parking, run:

```shell
python trainParking.py
```

To train others, specifying the env name in the args and run:

```shell
python train.py
```

The log file will store in dir "train/".

### 4.2 Eval

To eval the model, specify the env_name and model_time and run:

```shell
python train.py --env-name {} --model-time {}
```

The log file will store in dir "myeval/".

## 5 Result

## 6 Collaboration

This part is a record of project collaboration and is not important; you can skip it.

### 6.1 Understand the Highway Environment

We only focus on obs, action and reward function.

#### obs

For Occupancy grid observation(highway, intersection, and race track), variable obs is a three-dimensional array of 8\*11\*11. 8 stand for the number of feature of the obs.
`"features": ["presence", "on_road","x", "y", "vx", "vy", "cos_h", "sin_h"]`
11*11 is the size of view grid of our car. The feature of our car is located in the center and other is nearby vehicle's feature.

For Kinematics observation(parking), obs is a dict like：

``` shell
OrderedDict([
('observation', array([0, 0, 0, 0, 0, 1])),
('achieved_goal', array([0, 0, 0, 0, 0, 1])), 
('desired_goal', array([0, 0, 0, 0, 0, 1]))])
```

Each array stand for the features of the observation.
`"features": ["x", "y", "vx", "vy", "cos_h", "sin_h"]`

#### action

The action is config by:

```python
"action": {
    "type": "ContinuousAction",
    "longitudinal": True,
    "lateral": True,
},
```

Variable action is a array: [longitudinal speed, lateral speed]. Both speed is among [-1,1]. I don't firgure out how they realy stand for. But for an rl project, this is not important.
If we set longitudinal to false(for race track). Variable action is a array: [lateral speed].

### 6.2 Choose Algorithm

The task has a continous action space. So there are only 4 algorithm we can use.

- A2C/A3C
- DDPG
- TD3
- SAC
- PPO

The given file train.py is PPO algorithm. But as if it has some errors.

> We dicide to choose SAC for four env.

### 6.3 Code a Working SAC

We [SAC-continuous](https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/f0b32a5ce21af5f8620ee5b0201e284d9b009c24/8.SAC/SAC-continuous.py) and [cleanrl SAC](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)

### 6.4 Make Vector Environment

Todo

### 6.5 Attempts to Improve the Result

#### 6.5.1 Adjust the Size of Obs

We adjust the size of obs for highway, intersection and racetrack. This work very well for racetrack.

#### 6.5.2 Adjust the reward

Adjust the reward to 0 if the car don't work as expected.

#### 6.5.3 Reset the Environment

Reset the environment if the car don't work as expected, such as out of the way and run back.

## 7 Resources

### Web

1. [highwayEnv document](http://highway-env.farama.org/)
2. [highwayEnv code](https://github.com/Farama-Foundation/HighwayEnv)
3. [clearn rl -- a Deep Reinforcement Learning library](https://github.com/vwxyzjn/cleanrl/tree/master)
4. [highway by RainbowDQN -- A improved dqn algorithm](https://github.com/jackyoung96/RainbowDQN_highway) [^1]
5. [深度强化学习算法对比](https://zhuanlan.zhihu.com/p/342919579?utm_psn=1708635222873296896)

### Paper

## 8 Problem and Solution

Here are some problem we meet in the project.

### 8.1 X server error

Problem: 'GLIBCXX_3.4.30' not found for librosa in conda virtual environment

Solution: [STFW](https://bcourses.berkeley.edu/courses/1478831/pages/glibcxx-missing), we know this problem is caused by the version of GLIBCXX. The version of GLIBCXX in conda is 3.4.29, by the system require 3.4.30. So just search the available version in local environment and make a sysbolic link.

``` shell
cd FoldofVirtualEnvironmetLib
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```

[^1]:Based on a discrete action space, but ours is continuous.
