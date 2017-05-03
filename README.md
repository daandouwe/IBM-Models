# IBM-Models
Implementation of the alignment models IBM 1 and IBM 2 for the UvA course NLP2. Parameter estimation is performed using EM for the regular formulation of IBM 1 and 2, and Variational Inference for a Bayesian Formulation of IBM1. 

See the [project description](project1.pdf) for more details, and the [final report](report/final-report.pdf) for our findings.

## Alignments
See how the alignments change over 10 training epochs starting with uniform alignment probabilities. The width of the the line is proportional to its probability:

![sents0-4](sents/loaded/IBM2/uniform/sents-movie.gif)

Above we can see that the model predicts a perfect alignment at epoch 4. After this, the model unfortunately starts to wrongly align *le* to *program* instead of to the correct *the*.

The above results are from a uniformly initialized IBM model 2 running for 10 epochs, where each epoch is over the full 250k sentence dataset. The code is to produce these drawings is found in [util](lib/util.py). It was taken and adapted from a notebook in [this repository](https://github.com/INFR11133/lab1).

## Requirements
```
pip install tabulate
pip install progressbar2
```