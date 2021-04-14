# EE451 Final Project: Predicting Optimal Thread Count in Kernel Based Image Processing

This document is currently in progress, but will eventually contain the instructions for running the program and a brief explanation of how it works.

Compiling:

g++ main.cpp -o main `pkg-config --libs --cflags opencv` -lpthread

Running:

./main 1 ~/github/ee451-final-project/src/config.txt