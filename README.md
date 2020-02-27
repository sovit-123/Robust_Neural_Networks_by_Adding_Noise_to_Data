* # README

  

  ## 1) Research Paper Implementation

  * This project is an attempt to ***Building Robust Neural Network Models by Adding Noise to Image Data.***
  * The following are the research papers that I have tried the replicate the results and ideas from:
    * [**An empirical study on the effects of different types of noise in image classification tasks**](https://arxiv.org/pdf/1609.02781.pdf),  Gabriel B. Paranhos da Costa, Welinton A. Contato, Tiago S. Nazare, Jo ̃ao E. S. Batista Neto, Moacir Ponti.
    * [**Deep networks for robust visual recognition**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.170.1765&rep=rep1&type=pdf), Yichuan Tang, Chris Eliasmith.
    * [**Deep Convolutional Neural Networks and Noisy Images**](https://www.researchgate.net/publication/322915518_Deep_Convolutional_Neural_Networks_and_Noisy_Images), Tiago S. Nazare, Gabriel B. Paranhos da Costa, Welinton A. Contato, and Moacir Ponti (2018).

  

  ***Note:*** *I have included plots (inside `outputs/plots`) for both training files after training for 20 epochs. Take a loot at those for gaining faster insights into the project results.*

  

  ## 2) What is the Project About?

  * Neural networks are good at image recognition but are bad at handling noise. 
  * *So, to make them generalize better on noisy images, we can train them noisy images*.
  * And this project is an attempt to build robust image recognition neural networks by training them noisy data.

  

  ## 3) What Neural Network Model are We Using?

  * All of the training happens using the ResNet18 pre-trained models.
  * We are not using ImageNet weights, but are making all the hidden layer weights learnable.

  

  ## 4) Python Files Included and Ways to Execute Them

  * If the datasets are not present, then they will be downloaded to `inputs/data` directory. So, make sure the availability of internet connection before running any of the files.

  * All the executable python (`.py`) files are inside `src/` directory. All the python files can be executed from the command line. Different argument parsers are used for easy facilitation of training the neural networks.

  * There are three python files:

    * `add_noise.py`:

      * You can use this file to add ***gaussian, speckle, and salt & pepper noise*** to image data. *This file does not play any part in training of neural network models. Instead, the user can use this visualize how different types noise looks like.*

      * Execute the file:

        ```python
        # to add noise to CIFAR10 dataset
        python src/add_noise.py --dataset=cifar10 --gauss_noise=0.5 --salt_pep=0.5 --speckle_noise=0.5
        
        # to add noise to MNIST dataset
        python src/add_noise.py --dataset=mnist --gauss_noise=0.5 --salt_pep=0.5 --speckle_noise=0.5
        
        # to add noise to FashionMIST dataset
        python src/add_noise.py --dataset=fashionmnist --gauss_noise=0.5 --salt_pep=0.5 --speckle_noise=0.5
        ```

        

      * `--gauss_noise`, `--salt_pet`, `--speckle_noise` arguments define the amount of noise to add. They are optional arguments with default values already defined inside the python file.

      * All the preprocessing inside the file is done according to the dataset provided in the command line argument.

      * The resulting images will get stored inside `outputs/plots`.

    * `train_rnd_noise.py`:

      * You can execute this python file to train neural network model by applying ***random noise to image data***.

      * Execute the file:

        ```python
        # training without random noise, validating without random noise 
        python src/train_rnd_noise.py --epochs=20 --train_noise=no --test_noise=no
        
        # training with random noise, validating without random noise 
        python src/train_rnd_noise.py --epochs=20 --train_noise=yes --test_noise=no
        
        # training without random noise, validating with random noise 
        python src/train_rnd_noise.py --epochs=1 --train_noise=no --test_noise=yes
        
        # training with random noise, validating with random noise 
        python src/train_rnd_noise.py --epochs=1 --train_noise=yes --test_noise=yes
        ```

    * `train_gauss_noise.py`:

      * You can execute this python file to train neural network model by applying ***gaussian noise to image data***.

      * Execute the file:

        ```python
        # train with variance 0.5, validate with variance 0.5
        python src/train_gauss_noise.py --epochs=10 --train_noise=0.5 --test_noise=0.5
        ```

      * `--test_noise`: variance for validation images for the gaussian noise.

      * `--train_noise`: variance for training images for the gaussian noise.
