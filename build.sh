#!/bin/bash

nvcc -o train_cifar train_cifar.cu -lm -O3 -diag-suppress 1650