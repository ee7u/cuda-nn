Neural networks in C and cuda for learning purposes. The point is not to make a fancy model, or even an ok model. I aim to refresh my memory on the basics and to learn some cuda.

Some of the CPU code has been taken and modified from llm.c.

train_cifar.c contains a simple CPU implementation. train_cifar.cu contains a naive cuda implementation of the same model. Optimization is WIP, started by adding tiled matmuls (~2x faster than non-tiled).
