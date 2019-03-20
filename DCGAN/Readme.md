Implementation of [DCGAN](https://arxiv.org/abs/1511.06434.pdf) paper using pytorch framework on dogs dataset to generate dog like images from noise of dimension 64*64. The model is trained for 200 epochs. Training for 500-600 epochs will give good result.

# How to use:
1. Clone github repositories 
2. Open test.py
3. Enter number of images to generate (n variable)
4. Specify nrow and ncol in this file 
5. Run this code it will plot n images generated

# Samples generated:
![](https://github.com/sanchit2843/GAN-applications/blob/master/DCGAN/results/generated.png)

# Training Curve
![](https://github.com/sanchit2843/GAN-applications/blob/master/DCGAN/results/Generator_train.png)
# To do
Add argument parser for the test step
