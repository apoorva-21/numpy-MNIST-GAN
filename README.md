# numpy-MNIST-GAN

  An attempt at replicating an example I had read about and implemented in Keras for Adversarial Training of Generative and Discriminative Networks to generate MNIST digits, from scratch with numpy. I have tried to provide comments describing the algorithm along the way, with my attempt to implement it in code.


*In Dev: Activations of the Discriminator are thoroughly saturated due to Leaky ReLU. Trying to fix that to prevent vanishing gradient at the final sigmoid.


link to the blog post I referred to: https://skymind.ai/wiki/generative-adversarial-network-gan
link to the MNIST dataset(ubyte format, use script 'getImagesFromMNIST.py' to pickle the data):http://yann.lecun.com/exdb/mnist/
