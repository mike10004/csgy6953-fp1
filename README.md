# Transformers for Answering Crossword Clues

This is the codebase for my final project for CS-GY 6953 Deep Learning 
(Spring 2024 at NYU Tandon School of Engineering).

## Introduction

This codebase implements a transformer model for answering crossword clues.
Two models may be created and used: a word model that generates a word in response 
to a clue, and a letter model that generates a sequence of letters in response to a clue.

A report describing the models, their development, and evaluation is contained in the 
`report/output` directory.

The transformer implementation is adapted from https://github.com/chinmayhegde/dl-demos/blob/main/demo07-transformers.ipynb.

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
on meaning of the hyperparameters. As an example, the final models selected 
for evaluation in the report may be trained with these commands:

    $ python -m dlfp -m train -d onemark -p transformer_dropout_rate=0.0
    $ python -m dlfp -m train -d charmark -p emb_size=256

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

Other evaluation configuration parameters may be set with the `-e` or `--eval-config`
option. For example, to define an alternative beam search strategy, use 
`--eval-config max_ranks=5,4,3,2,1`.

The evaluation command creates a CSV file that has a row for every clue/answer pair,
showing what the top suggested answers were and where the actual answer ranks among
the suggestions.

## Code

The modules are as follows:

* **dlfp**
  + **common**: methods for general use (e.g. file I/O, timestamps)
  + **utils**: classes and methods relating to language concepts, e.g. tokenization and vocabularies
  + **datasets**: classes and methods relating to dataset loading and manipulation
  + **models**: model and hyperparameter code; this is where the *Cruciformer* model is defined
  + **train**: training code
  + **translate**: sequence generator code (including beam search)
  + **running**, **main**: command line interface implementation
  + **results**: code for analyzing results, e.g. generating accuracy tables from sequence generation CSV files
  + **baseline**
    - **cs** [Crossword-Solver](https://github.com/pncnmnp/Crossword-Solver) baseline implementation
* **dlfp_tests**: unit tests; probably not interesting unless you're really tinkering

## Results Visualization

The notebook `dlfp/nb/figures.ipynb` may be used to generate plots of loss curves 
and evaluation accuracy. Change the pathnames to refer to wherever your results files
are stored.