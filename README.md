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
.
├── README.md
├── TD3 \\TD3 algorithm
│   ├── TD3v0
│   │   ├── DDPG.py
│   │   ├── OurDDPG.py
│   │   ├── README.md
│   │   ├── TD3.py
│   │   ├── main.py
│   │   ├── run_experiments.sh
│   │   └── utils.py
│   ├── TD3v1
│   │   ├── TD3.py
│   │   ├── huggingface.py
│   │   └── td3_eval.py
│   └── TD3v2
│       └── TD3.py
├── demo \\demo by sb3
│   ├── SAC_sb3_parking_demo.py
│   ├── SAC_sb3_racetrack_demo.py
│   ├── TD3_sb3_parking_demo.py
│   └── TD3_sb3_racetrack_demo.py
├── eval
│   ├── eval_highway.py
│   ├── eval_intersection.py
│   ├── eval_parking.py
│   └── eval_racetrack.py
├── requirements.txt
└── train.py
```

## 4 Running Guide

## 5 Result

## 6 Collaboration

This part is a record of project collaboration and is not important; you can skip it.

### 6.1 Understand the Highway Environment (by Gonghu Shang)

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

### Choose Algorithm

The task has a continous action space. So there are only 4 algorithm we can use.

- A2C/A3C
- DDPG
- TD3
- SAC
- PPO

The given file train.py is PPO algorithm. But as if it has some errors.

I decide to choose TD3 for a test. TD3 is original implementation. TD3v1 and Td3v2 is other implementations I found. But another problem is some of them are using gym instead of gymnasium in the code.

## 7 Resources

### Web

1. [highwayEnv document](http://highway-env.farama.org/)
2. [highwayEnv code](https://github.com/Farama-Foundation/HighwayEnv)
3. [clearn rl -- a Deep Reinforcement Learning library](https://github.com/vwxyzjn/cleanrl/tree/master)
4. [highway by RainbowDQN -- A improved dqn algorithm](https://github.com/jackyoung96/RainbowDQN_highway) [^1]

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
