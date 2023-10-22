# Ch. 5: Computer Vision
This is the assignment from Chapter 5: Computer Vision.

In this chapter, I do:
- Implementation of digital image processing to improve image quality using Pooling and CLAHE techniques.
- Implementation of the Transfer Learning method for multi-classification cases using Deep Learning-based Computer Vision modeling, such as ResNet, DenseNet, and Vision Transformer (ViT).

## Task 1
### Case Study Description
As of July 2023, both Apple and Samsung lead the smartphone industry worldwide, with a combined 52.61% of the total market share [(ref1)](https://www.oberlo.com/statistics/smartphone-market-share). As the main feature that must be present on today's smartphones, Apple and Samsung are competing to create camera technology so you can capture your best photo even in the low light condition.
- In September 2019, Apple introduced **Deep Fusion** technology (via the iPhone 11 series) to tackle the challenge. Its upgrade, named **Photonic Engine**, was introduced in September 2022 via the new iPhone 14 series [(ref2)](https://www.youtube.com/watch?v=ux6zXguiqxM&t=4784s&ab_channel=Apple).
- In February 2023, Samsung introduced **Adaptive Tetra-squared Pixel Sensor** technology with the Samsung S23 series as a counter-solution to a similar problem, promising excellent bright photo results from dark-toned images [(ref3)](https://www.youtube.com/watch?v=gUM2wYKdxDA&t=742s&ab_channel=Samsung).

At its core, both technologies work by combining several adjacent pixels into a single pixel, using a **Max Pooling** operation. In this case, you are challenged to replicate the concept (brighten dark-toned photos), and then compare the result with another approach, i.e., **Contrast Limited Adaptive Histogram Equation (CLAHE)**.

## Task 2
### Case Study Description
A new robotic facility located in East Kalimantan, near the Titik Nol Ibu Kota Negara (IKN) Indonesia, asks you to create a Computer Vision model for their new droid (robot) products. The company requests you to **teach the robot how to read a sequence of numbers**. You suddenly realize that the first stage is to let the robot correctly identify each individual digit (0-9). However, since the prototype announcement date was hastened, your deadline is very tight: you only have **less than 1 week** to complete the job. As a professional AI developer, you keep calm and know that you can exploit the **Transfer Learning** method to solve this problem efficiently.

As a basic dataset in most of Computer Vision tasks, **Modified National Institute of Standards and Technology (MNIST) database** contains 10 handwritten digits. All of them are in the grayscale (1-channel). Torchvision, a sub-library of PyTorch, has dozens of pre-trained models that you can easily choose from. All of these models were originally trained on the ImageNet dataset [(ref1)](https://www.image-net.org/download.php), which contains millions of RGB (3-channel) images and 1,000 classes. For simplicity, let choose **Resnet18** [(ref2)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), **DenseNet121** [(ref3)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf), and **Vision Transformer (ViT)** [(ref4)](https://arxiv.org/pdf/2010.11929.pdf) as baseline, state-of-the-art models to test the **image classification** performance. Your complete tasks are as follows.

1. Pick **DenseNet** as your first model to experiment with, then **change the number of neurons in the first and last layers** (since the ImageNet has 1,000 classes, while MNIST only has 10 classes; both are also come with different image size and channel).
2. Define **hyperparameters** and train the model (all **layers are trainable**).
3. Plot the model performance, for both **training** and **validation** results.
4. Now try to **freeze (layers are non-trainable) some parts** of layers: (1) "denseblock1", (2) "denseblock1" and "denseblock2". These will be two separate models.
5. **Retrain** each model, plot its performance, and examine the difference.
6. BONUS: Can you **replicate** all of the steps above with different models, i.e., **ResNet** and **ViT**?

## Explain
In Task 1, I worked with dark photos and trying to make them brighter using different methods: Max Pooling, Min Pooling, and Average Pooling. We'll also compare these methods to another one called CLAHE, which enhances image contrast.

In Task 2, I created a machine learning model, specifically a Convolutional Neural Network, to predict handwritten digits from the MNIST dataset. We'll do this in two steps: first, we'll train the model with some layers frozen, and then we'll unfreeze those layers and retrain the model. We'll visualize and compare how well these models perform and see if freezing layers affects the results. As a bonus, we'll repeat all these steps with different models like ResNet and ViT.
