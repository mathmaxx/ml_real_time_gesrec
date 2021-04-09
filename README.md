#  Hand gesture recognition

[![License](http://img.shields.io/badge/license-MIT-green.svg?style=flat)](https://github.com/Solvve/ml_job_classifier/blob/master/LICENSE.txt)

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