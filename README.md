# IBM-Models
Implementation of the alignment models IBM 1 and IBM 2 for the UvA course [NLP2](https://uva-slpl.github.io/nlp2/). Parameter estimation is performed using EM for the regular formulation of IBM 1 and 2, and Variational Inference for a Bayesian Formulation of IBM1. Joint work with [Fije van Overeem](https://github.com/Fije) and [Tim van Elsloo](https://github.com/elslooo).

See the [project description](project1.pdf) for more details, and the [final report](report/final-report.pdf) for our findings.

## Alignments
See how the alignments change over 10 epochs of training. The width of each line is proportional to its probability, and we start with uniform alignment probabilities:

![sents0-4](sents/loaded/IBM2/uniform/sents-movie.gif)

These predictions are from an IBM model 2 running for 10 epochs, where each epoch is over the full 250k sentence dataset. We can see that the model predicts a perfect alignment at epoch 4. After this, the model unfortunately starts to wrongly align *le* to *has* instead of to the correct *the*.

The code is to produce these drawings is found in [util](lib/util.py). It was taken and adapted from a notebook in [this repository](https://github.com/INFR11133/lab1).

## Requirements
```
pip install tabulate
pip install progressbar2
```