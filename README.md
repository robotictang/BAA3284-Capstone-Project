# BAA3284: Capstone Project 
Python and Applications of Deep Neural Networks

[Sunway University](https://university.sunway.edu.my/)

Instructor: [Dr Tang Tiong Yew](https://university.sunway.edu.my/profiles/subs/Dr-Tang-Tiong-Yew)

**The content of this course changes as technology evolves**, to keep up to date with changes [follow me on GitHub](https://github.com/robotictang).

# Course Description

[Overview Video](https://www.youtube.com/watch?v=ZLiRyVt4bWg)

Deep learning is a group of exciting new technologies for neural networks. Through a combination of advanced training techniques and neural network architectural components, it is now possible to create neural networks that can handle tabular data, images, text, and audio as both input and output. Deep learning allows a neural network to learn hierarchies of information in a way that is like the function of the human brain. This course will introduce the student to classic neural network structures, Convolution Neural Networks (CNN), Long Short-Term Memory (LSTM), Gated Recurrent Neural Networks (GRU), General Adversarial Networks (GAN) and reinforcement learning. Application of these architectures to computer vision, time series, security, natural language processing (NLP), and data generation will be covered. High Performance Computing (HPC) aspects will demonstrate how deep learning can be leveraged both on graphical processing units (GPUs), as well as grids. Focus is primarily upon the application of deep learning to problems, with some introduction to mathematical foundations. Students will use the Python programming language to implement deep learning using Google TensorFlow and Keras. It is not necessary to know Python prior to this course; however, familiarity of at least one programming language is assumed. This course will be delivered in a hybrid format that includes both classroom and online instruction.

# Textbook

The original author of this code was contributed by Jeff Heaton and the code is then adapted to be used in this course with Apache License Version 2.0. This same material is also available in [book format](https://www.heatonresearch.com/book/applications-deep-neural-networks-keras.html). The course textbook is “Applications of Deep Neural networks with Keras“, ISBN 9798416344269.

If you would like to cite the material from this course/book, please use the following BibTex citation:

```
@misc{heaton2020applications,
    title={Applications of Deep Neural Networks},
    author={Jeff Heaton},
    year={2020},
    eprint={2009.05673},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

# Objectives

1. Explain how neural networks (deep and otherwise) compare to other machine learning models.
2. Determine when a deep neural network would be a good choice for a particular problem.
3. Demonstrate your understanding of the material through a capstone project uploaded to GitHub.

Module|Content
---|---
[Module 1](t81_558_class_01_1_overview.ipynb)<br> | **Module 1: Python Preliminaries**<ul><li>Part 1.1: Course Overview<li>Part 1.2: Introduction to Python<li>Part 1.3: Python Lists, Dictionaries, Sets & JSON<li>Part 1.4: File Handling<li>Part 1.5: Functions, Lambdas, and Map/ReducePython Preliminaries</ul>
[Module 2](t81_558_class_02_1_python_pandas.ipynb)<br> | **Module 2: Python for Machine Learning**<ul><li>	Part 2.1: Introduction to Pandas for Deep Learning<li>Part 2.2: Encoding Categorical Values in Pandas<li>Part 2.3: Grouping, Sorting, and Shuffling<li>Part 2.4: Using Apply and Map in Pandas<li>Part 2.5: Feature Engineering in Padas</ul>
[Module 3](t81_558_class_03_1_neural_net.ipynb)<br> | **Module 3: TensorFlow and Keras for Neural Networks**<ul><li>Part 3.1: Deep Learning and Neural Network Introduction<li>Part 3.2: Introduction to Tensorflow & Keras<li>Part 3.3: Saving and Loading a Keras Neural Network<li>Part 3.4: Early Stopping in Keras to Prevent Overfitting<li>Part 3.5: Extracting Keras Weights and Manual Neural Network Calculation</ul>
[Module 4](t81_558_class_04_1_feature_encode.ipynb)<br> |**Module 4: Training for Tabular Data**<ul><li>Part 4.1: Encoding a Feature Vector for Keras Deep Learning<li>Part 4.2: Keras Multiclass Classification for Deep Neural Networks with ROC and AUC<li>Part 4.3: Keras Regression for Deep Neural Networks with RMSE<li>Part 4.4: Backpropagation, Nesterov Momentum, and ADAM Training<li>Part 4.5: Neural Network RMSE and Log Loss Error Calculation from Scratch</ul>
[Module 5](t81_558_class_05_1_reg_ridge_lasso.ipynb)<br> | **Module 5: Regularization and Dropout**<ul><li>Part 5.1: Introduction to Regularization: Ridge and Lasso<li>Part 5.2: Using K-Fold Cross Validation with Keras<li>Part 5.3: Using L1 and L2 Regularization with Keras to Decrease Overfitting<li>Part 5.4: Drop Out for Keras to Decrease Overfitting<li>Part 5.5: Bootstrapping and Benchmarking Hyperparameters</ul>
[Module 6](t81_558_class_06_1_python_images.ipynb)<br> | **Module 6: CNN for Vision**<ul>	Part 6.1: Image Processing in Python<li>Part 6.2: Using Convolutional Networks with Keras<li>Part 6.3: Using Pretrained Neural Networks<li>Part 6.4: Looking at Keras Generators and Image Augmentation<li>Part 6.5: Recognizing Multiple Images with YOLOv5</ul>
[Module 7](t81_558_class_07_1_gan_intro.ipynb)<br> | **Module 7: Generative Adversarial Networks (GANs)**<ul><li>Part 7.1: Introduction to GANS for Image and Data Generation<li>Part 7.2: Train StyleGAN3 with your Own Images<li>Part 7.3: Exploring the StyleGAN Latent Vector<li>Part 7.4: GANS to Enhance Old Photographs Deoldify<li>Part 7.5: GANs for Tabular Synthetic Data Generation</ul>
[Module 8](t81_558_class_08_1_kaggle_intro.ipynb)<br> | **Module 8: Kaggle**<ul><li>Part 8.1: Introduction to Kaggle<li>Part 8.2: Building Ensembles with Scikit-Learn and Keras<li>Part 8.3: How Should you Architect Your Keras Neural Network: Hyperparameters<li>Part 8.4: Bayesian Hyperparameter Optimization for Keras<li>Part 8.5: Current Semester's Kaggle</ul>
[Module 9](t81_558_class_09_1_keras_transfer.ipynb)<br> | **Module 9: Transfer Learning**<ul><li>Part 9.1: Introduction to Keras Transfer Learning<li>Part 9.2: Keras Transfer Learning for Computer Vision<li>Part 9.3: Transfer Learning for NLP with Keras<li>Part 9.4: Transfer Learning for Facial Feature Recognition<li>Part 9.5: Transfer Learning for Style Transfer</ul>
[Module 10](t81_558_class_10_1_timeseries.ipynb)<br> | **Module 10: Time Series in Keras**<ul><li>Part 10.1: Time Series Data Encoding for Deep Learning, Keras<li>Part 10.2: Programming LSTM with Keras and TensorFlow<li>Part 10.3: Text Generation with Keras<li>Part 10.4: Introduction to Transformers<li>Part 10.5: Transformers for Timeseries</ul>
[Module 11](t81_558_class_11_01_huggingface.ipynb)<br> | **Module 11: Natural Language Processing**<ul><li>Part 11.1: Hugging Face Introduction<li>Part 11.2: Hugging Face Tokenizers<li>Part 11.3: Hugging Face Data Sets<li>Part 11.4: Training a Model in Hugging Face<li>Part 11.5: What are Embedding Layers in Keras</ul>
[Module 12](t81_558_class_12_01_ai_gym.ipynb)<br> | **Module 12: Reinforcement Learning**<ul><li>Kaggle Assignment due: 11/29/2022 (approx 4-6PM, due to Kaggle GMT timezone)<li>Part 12.1: Introduction to the OpenAI Gym<li>Part 12.2: Introduction to Q-Learning for Keras<li>Part 12.3: Keras Q-Learning in the OpenAI Gym<li>Part 12.4: Atari Games with Keras Neural Networks<li>Part 12.5: Application of Reinforcement Learning</ul>
[Module 13](t81_558_class_13_01_flask.ipynb)<br> | **Module 13: Deployment and Monitoring**<ul><li>Part 13.1: Flask and Deep Learning Web Services <li>Part 13.2: Interrupting and Continuing Training<li>Part 13.3: Using a Keras Deep Neural Network with a Web Application<li>Part 13.4: When to Retrain Your Neural Network<li>Part 13.5: Tensor Processing Units (TPUs)</ul>

# Datasets

* [Datasets can be downloaded here](https://data.heatonresearch.com/data/t81-558/index.html)

# Videos

**Fundamentals of Deep learning**

[Part 1](https://www.youtube.com/watch?v=aircAruvnKk)

[Part 2](https://www.youtube.com/watch?v=IHZwWFHWa-w)

[Part 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

[Part 4](https://www.youtube.com/watch?v=tIeHLnjs5U8)

**Previous LSTM Approaches**

[LSTM](https://www.youtube.com/watch?v=QciIcRxJvsM)

**State-of-the-art Transformers Approaches**

[Part 1](https://www.youtube.com/watch?v=TQQlZhbC5ps)

[Part 2](https://www.youtube.com/watch?v=xI0HHN5XKDo)

[Part 3](https://www.youtube.com/watch?v=BGKumht1qLA)

**CPU VS GPU VS TPU (Training Speed Test)**

[Video](https://www.youtube.com/watch?v=6TOojA72uo4) Summary: CPU: 45 min , GPU: 55 sec, TPU: 15sec

**Google Colab VS AWS SageMaker Studio Lab** 

[Video](https://www.youtube.com/watch?v=qFH1MV-yg04&t=350s)

**Tips to Use Colab**

[Video](https://www.youtube.com/watch?v=qWVeJ7Esx2U)

