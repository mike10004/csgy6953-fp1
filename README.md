# Transformers for Answering Crossword Clues

This is the codebase for my final project for CS-GY 6953 Deep Learning 
(Spring 2024 at NYU Tandon School of Engineering).

## Introduction

This codebase includes implementations of transformers for answering crossword clues.
Two models may be created and used: a word model that generates a word in response 
to a clue, and a letter model that generates a sequence of letters in response to a clue.

## Prerequisites

* Download the dataset using the instructions here: https://github.com/text-machine-lab/xword_benchmark
  + Unzip the dataset to `data/datasets/benchmark`
* Install requirements with `pip install -r requirements.txt`

## Command line interface

Several modules are runnable by invoking `python -m <module>`. Each provides
a help message if the `--help` option is present.

### Data preparation

Create the `onemark` and `charmark` datasets for training and evaluating the word and letter models.

    $ python -m dlfp.datasets -m create -d onemark
    $ python -m dlfp.datasets -m create -d charmark

The datasets are created using *benchmark* as a source. They are written to the `data/datasets` directory. 

### Training

Execute the following commands to train the database.

    $ python -m dlfp -m train -d onemark     # train word model
    $ python -m dlfp -m train -d charmark    # train letter model

Use the `--train-param` and `--model-param` options to set hyperparameters. 
For example, `--train-param lr=0.001` sets the learning rate to 0.001.
See the `TrainHyperparametry` and `ModelHyperparametry` classes for details 
on meaning of the hyperparameters.

When training is finished, a checkpoint file is written to a directory 
beneath `$PWD/checkpoints`. The pathname of this file is necessary for 
the testing command.

### Testing

Sequence generation can be an expensive operation, so you may want to create 
smaller subsets for testing.

    # create a 1000-pair subset of the validation set
    $ python -m dlfp.datasets -m subset -d onemark --split valid --shuffle 123 --size 1000

This creates a dataset called `valid_r123_s1000` in the `datasets/onemark` directory.

Execute evaluation with the following command:

    $ python -m dlfp -m eval -d onemark -e split=valid_r123_s1000 -f $CHECKPOINT_FILE

Replace `$CHECKPOINT_FILE` with the pathname of the checkpoint file created by the 
training command.

## Results Visualization

The notebook `dlfp/nb/figures.ipynb` may be used to generate plots of loss curves 
and evaluation accuracy.