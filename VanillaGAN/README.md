**Description** <br>
This is an implementation of a Generative Adversarial Network from Ian Goodfellow's paper (<https://arxiv.org/pdf/1406.2661.pdf>), where the Generator and Discriminator are described by multilayer perceptrons. Several empirical techniques have been implemented to improve stability: <br>

  - One-sided label smoothing: The discriminator has been penalised when the prediction for any real images goes beyond 0.9, by setting our target label value to be 0.9 instead of 1.0. 
  - The Adam optimiser is used instead of SGD with momentum, to accelerate convergence and prevent oscillations in local minima
  - Noisy images are spherical instead of linear, by sampling from gaussian rather than a uniform distribution
  - The image pixels are scaled between -1 and 1, and tanh is used as the output layer for the generator
  - Batch normalisation is used to further stabilise training

For more tricks, visit <https://github.com/soumith/ganhacks>

**Key result**  <br>
GIF of sample images over 200 epochs and 2h training:

![Alt Text](https://github.com/christinakouridi/TensorFlow2.0/blob/master/VanillaGAN/GAN.gif)


