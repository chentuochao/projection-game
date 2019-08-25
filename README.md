# projection-game
Created by PKU Geeklab

*A small project for summer recreation*

## Introduction
This game needs a *projector* and a *camera*. 

It is intended to use a *projector* to project several different shapes to a wall and plays some notes when player puts his/her hand on the projected shape.

The *camera* is used for video detection of the player's hands.

## Installation
To play this game, you need to install a [human pose detection model](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git), and add the path of the model to your `sys.path`, and then follow the instruction of the repo to download a [pretrained model weights file](https://github.com/marvis/pytorch-mobilenet) under the repo dir.

PyTorch, OpenCV is also required.
```bash
# go somewhere
git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git
# go to this repo
cd <projection_game-dir>
python game.py --package-dir <lightweight-human-pose-estimation.pytorch.git-dir> --checkpoint-dir <lightweight-human-pose-estimation.pytorch.git/checkpoint-file> --video 0
```
If there isn't any gpu in your machine, append `--cpu` option to the last command above.
