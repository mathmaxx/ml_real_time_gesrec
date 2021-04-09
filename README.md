#  Hand gesture recognition

[![License](http://img.shields.io/badge/license-MIT-green.svg?style=flat)](https://github.com/Solvve/ml_job_classifier/blob/master/LICENSE.txt)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![scikit-learn 0.23.2](https://img.shields.io/badge/scikit_learn-0.24.1-blue)](https://scikit-learn.org/stable/)
[![pandas 1.2.3](https://img.shields.io/badge/pandas-1.2.3-blue)](https://pypi.org/project/pandas/)
[![numpy 1.20.2](https://img.shields.io/badge/numpy-1.20.2-blue)](https://numpy.org/install/)
[![torch 0.23.2](https://img.shields.io/badge/torch-1.8.1-blue)](https://pytorch.org/)
[![Solvve](https://img.shields.io/badge/made%20in-solvve-blue)](https://solvve.com/)


## Description

This is an example of hand recognition gesture task.

The main idea it was add new gesture class without retrain whole model or head layers.
For solving this problem we follow the next steps:
* Remove fc and softmax
* Generate a vector representation for each gesture in train

Forked from https://github.com/ahmetgunduz/Real-time-GesRec.

To classify vectors we training 3 different classifiers with comparing results and training time:

| Classifier | Accuracy | Time (sec)|
|---|---|---|
| KNN | 0.89 | 108 |
| NC + KNN | 0.87 | 1.95 |
| SVC | 0.9 | 1665 |

Research in notebook -> embedding_representations.ipynb
Demo script with camera capture -> simple_online_demo.py

Example demo:

![](gifs/demo.gif)

