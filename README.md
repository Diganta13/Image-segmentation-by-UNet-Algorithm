# Image-segmentation-by-UNet-Algorithm

1. Introduction

Computer vision is an interdisciplinary scientific field that deals with how computers can be made to gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to automate tasks that the human visual system can do. (Wikipedia)
CV is a very interdisciplinary field

Deep Learning has enabled the field of Computer Vision to advance rapidly in the last few years. In this post I would like to discuss about one specific task in Computer Vision called as Semantic Segmentation. Even though researchers have come up with numerous ways to solve this problem, I will talk about a particular architecture namely UNET, which use a Fully Convolutional Network Model for the task.

We will use UNET to build a first-cut solution to the TGS Salt Identification challenge hosted by Kaggle.

Along with this, my purpose of writing the blog is to also provide some intuitive insights on the commonly used operations and terms in Convolutional Networks for Image understanding. Some of these include Convolution, Max Pooling, Receptive field, Up-sampling, Transposed Convolution, Skip Connections, etc.
2. Prerequisites

I will assume that the reader is already familiar with the basic concepts of Machine Learning and Convolutional Networks. Also you must have some working knowledge of ConvNets with Python and Keras library.
3. What is Semantic Segmentation?

There are various levels of granularity in which the computers can gain an understanding of images. For each of these levels there is a problem defined in the Computer Vision domain. Starting from a coarse grained down to a more fine grained understanding, let’s describe these problems below:
a. Image classification
Image Classification

The most fundamental building block in Computer Vision is the Image classification problem where given an image, we expect the computer to output a discrete label, which is the main object in the image. In image classification we assume that there is only one (and not multiple) object in the image.
b. Classification with Localization
Classification with localization

In localization along with the discrete label, we also expect the compute to localize where exactly the object is present in the image. This localization is typically implemented using a bounding box which can be identified by some numerical parameters with respect to the image boundary. Even in this case, the assumption is to have only one object per image.
c. Object Detection
Object Detection

Object Detection extends localization to the next level where now the image is not constrained to have only one object, but can contain multiple objects. The task is to classify and localize all the objects in the image. Here again the localization is done using the concept of bounding box.
d. Semantic Segmentation
Semantic Segmentation

The goal of semantic image segmentation is to label each pixel of an image with a corresponding class of what is being represented. Because we’re predicting for every pixel in the image, this task is commonly referred to as dense prediction.

Note that unlike the previous tasks, the expected output in semantic segmentation are not just labels and bounding box parameters. The output itself is a high resolution image (typically of the same size as input image) in which each pixel is classified to a particular class. Thus it is a pixel level image classification.
e. Instance segmentation
Instance Segmentation

Instance segmentation is one step ahead of semantic segmentation wherein along with pixel level classification, we expect the computer to classify each instance of a class separately. For example in the image above there are 3 people, technically 3 instances of the class “Person”. All the 3 are classified separately (in a different color). But semantic segmentation does not differentiate between the instances of a particular class.

If you are still confused between the differences of object detection, semantic segmentation and instance segmentation, below image will help to clarify the point:
Object Detection vs Semantic Segmentation vs Instance Segmentation

In this post we will learn to solve the Semantic Segmentation problem using Fully Convolutional Network (FCN) called UNET.
4. Applications

If you are wondering, whether semantic segmentation is even useful or not, your query is reasonable. However, it turns out that a lot of complex tasks in Vision require this fine grained understanding of images. For example:
a. Autonomous vehicles

Autonomous driving is a complex robotics tasks that requires perception, planning and execution within constantly evolving environments. This task also needs to be performed with utmost precision, since safety is of paramount importance. Semantic Segmentation provides information about free space on the roads, as well as to detect lane markings and traffic signs.
Source: https://www.youtube.com/watch?v=ATlcEDSPWXY
b. Bio Medical Image Diagnosis

Machines can augment analysis performed by radiologists, greatly reducing the time required to run diagnostic tests.
Source: https://arxiv.org/abs/1701.08816
c. Geo Sensing

Semantic Segmentation problems can also be considered classification problems, where each pixel is classified as one from a range of object classes. Thus, there is a use case for land usage mapping for satellite imagery. Land cover information is important for various applications, such as monitoring areas of deforestation and urbanization.

To recognize the type of land cover (e.g., areas of urban, agriculture, water, etc.) for each pixel on a satellite image, land cover classification can be regarded as a multi-class semantic segmentation task. Road and building detection is also an important research topic for traffic management, city planning, and road monitoring.

There are few large-scale publicly available datasets (Eg : SpaceNet), and data labeling is always a bottleneck for segmentation tasks.
Source: https://blog.playment.io/semantic-segmentation/
d. Precision Agriculture

Precision farming robots can reduce the amount of herbicides that need to be sprayed out in the fields and semantic segmentation of crops and weeds assist them in real time to trigger weeding actions. Such advanced image vision techniques for agriculture can reduce manual monitoring of agriculture.
Source: https://blog.playment.io/semantic-segmentation/

We will also consider a practical real world case study to understand the importance of semantic segmentation. The problem statement and the datasets are described in the below sections.
5. Business Problem

In any Machine Learning task, it is always suggested to spend a decent amount of time in aptly understanding the business problem that we aim to solve. This not only helps to apply the technical tools efficiently but also motivates the developer to use his/her skills in solving a real world problem.

TGS is one of the leading Geo-science and Data companies which uses seismic images and 3D renderings to understand which areas beneath the Earth’s surface which contain large amounts of oil and gas.

Interestingly, the surfaces which contain oil and gas, also contain huge deposits of salt. So with the help of seismic technology, they try to predict which areas in the surface of the Earth contain huge amount of salts.

Unfortunately, professional seismic imaging requires expert human vision to exactly identify salt bodies. This leads to highly subjective and variable renderings. Moreover it could cause huge loss for the oil and gas company drillers if the human prediction is incorrect.

Thus TGS hosted a Kaggle Competition, to employ machine vision to solve this task with better efficiency and accuracy.

To read more about the challenge, click here.

To read more about seismic technology, click here.
6. Understanding the data

Download the data files from here.

For simplicity we will only use train.zip file which contains both the images and their corresponding masks.

In the images directory, there are 4000 seismic images which are used by human experts to predict whether there could be salt deposits in that region or not.

In the masks directory, there are 4000 gray scale images which are the actual ground truth values of the corresponding images which denote whether the seismic image contains salt deposits and if so where. These will be used for building a supervised learning model.

Let’s visualize the given data to get a better understanding:
Sample data point and corresponding label

The image on left is the seismic image. The black boundary is drawn just for the sake of understanding denoting which part contains salt and which does not. (Of course this boundary is not a part of the original image)

The image on the right is called as the mask which is the ground truth label. This is what our model must predict for the given seismic image. The white region denotes salt deposits and the black region denotes no salt.

Let’s look at a few more images:
Sample data point and corresponding label
Sample data point and corresponding label
Sample data point and corresponding label

Notice that if the mask is entirely black, this means there are no salt deposits in the given seismic image.

Clearly from the above few images it can be inferred that its not easy for human experts to make accurate mask predictions for the seismic images.
7. Understanding Convolution, Max Pooling and Transposed Convolution

Before we dive into the UNET model, it is very important to understand the different operations that are typically used in a Convolutional Network. Please make a note of the terminologies used.
i. Convolution operation

There are two inputs to a convolutional operation

i) A 3D volume (input image) of size (nin x nin x channels)

ii) A set of ‘k’ filters (also called as kernels or feature extractors) each one of size (f x f x channels), where f is typically 3 or 5.

The output of a convolutional operation is also a 3D volume (also called as output image or feature map) of size (nout x nout x k).

The relationship between nin and nout is as follows:
Convolution Arithmetic

Convolution operation can be visualized as follows:
Source: http://cs231n.github.io/convolutional-networks/

In the above GIF, we have an input volume of size 7x7x3. Two filters each of size 3x3x3. Padding =0 and Strides = 2. Hence the output volume is 3x3x2. If you are not comfortable with this arithmetic then you need to first revise the concepts of Convolutional Networks before you continue further.

One important term used frequently is called as the Receptive filed. This is nothing but the region in the input volume that a particular feature extractor (filter) is looking at. In the above GIF, the 3x3 blue region in the input volume that the filter covers at any given instance is the receptive field. This is also sometimes called as the context.

To put in very simple terms, receptive field (context) is the area of the input image that the filter covers at any given point of time.
ii) Max pooling operation

In simple words, the function of pooling is to reduce the size of the feature map so that we have fewer parameters in the network.

For example:
Source: https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks#

Basically from every 2x2 block of the input feature map, we select the maximum pixel value and thus obtain a pooled feature map. Note that the size of the filter and strides are two important hyper-parameters in the max pooling operation.

The idea is to retain only the important features (max valued pixels) from each region and throw away the information which is not important. By important, I mean that information which best describes the context of the image.

A very important point to note here is that both convolution operation and specially the pooling operation reduce the size of the image. This is called as down sampling. In the above example, the size of the image before pooling is 4x4 and after pooling is 2x2. In fact down sampling basically means converting a high resolution image to a low resolution image.

Thus before pooling, the information which was present in a 4x4 image, after pooling, (almost) the same information is now present in a 2x2 image.

Now when we apply the convolution operation again, the filters in the next layer will be able to see larger context, i.e. as we go deeper into the network, the size of the image reduces however the receptive field increases.

For example, below is the LeNet 5 architecture:
LeNet 5

Notice that in a typical convolutional network, the height and width of the image gradually reduces (down sampling, because of pooling) which helps the filters in the deeper layers to focus on a larger receptive field (context). However the number of channels/depth (number of filters used) gradually increase which helps to extract more complex features from the image.

Intuitively we can make the following conclusion of the pooling operation. By down sampling, the model better understands “WHAT” is present in the image, but it loses the information of “WHERE” it is present.
iii) Need for up sampling

As stated previously, the output of semantic segmentation is not just a class label or some bounding box parameters. In-fact the output is a complete high resolution image in which all the pixels are classified.

Thus if we use a regular convolutional network with pooling layers and dense layers, we will lose the “WHERE” information and only retain the “WHAT” information which is not what we want. In case of segmentation we need both “WHAT” as well as “WHERE” information.

Hence there is a need to up sample the image, i.e. convert a low resolution image to a high resolution image to recover the “WHERE” information.

In the literature, there are many techniques to up sample an image. Some of them are bi-linear interpolation, cubic interpolation, nearest neighbor interpolation, unpooling, transposed convolution, etc. However in most state of the art networks, transposed convolution is the preferred choice for up sampling an image.
iv) Transposed Convolution

Transposed convolution (sometimes also called as deconvolution or fractionally strided convolution) is a technique to perform up sampling of an image with learnable parameters.

I will not describe how transpose convolution works because Naoki Shibuya has already done a brilliant job in his blog Up sampling with Transposed Convolution. I strongly recommend you to go through this blog (multiple times if required) to understand the process of Transposed Convolution.

However, on a high level, transposed convolution is exactly the opposite process of a normal convolution i.e., the input volume is a low resolution image and the output volume is a high resolution image.

In the blog it is nicely explained how a normal convolution can be expressed as a matrix multiplication of input image and filter to produce the output image. By just taking the transpose of the filter matrix, we can reverse the convolution process, hence the name transposed convolution.
v) Summary of this section

After reading this section, you must be comfortable with following concepts:

    Receptive field or context
    Convolution and pooling operations down sample the image, i.e. convert a high resolution image to a low resolution image
    Max Pooling operation helps to understand “WHAT” is there in the image by increasing the receptive field. However it tends to lose the information of “WHERE” the objects are.
    In semantic segmentation it is not just important to know “WHAT” is present in the image but it is equally important to know “WHERE” it is present. Hence we need a way to up sample the image from low resolution to high resolution which will help us restore the “WHERE” information.
    Transposed Convolution is the most preferred choice to perform up sampling, which basically learns parameters through back propagation to convert a low resolution image to a high resolution image.

If you are confused with any of the terms or concepts explained in this section, feel free to read it again till you get comfortable.
8. UNET Architecture and Training

The UNET was developed by Olaf Ronneberger et al. for Bio Medical Image Segmentation. The architecture contains two paths. First path is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is just a traditional stack of convolutional and max pooling layers. The second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using transposed convolutions. Thus it is an end-to-end fully convolutional network (FCN), i.e. it only contains Convolutional layers and does not contain any Dense layer because of which it can accept image of any size.

In the original paper, the UNET is described as follows:

If you did not understand, its okay. I will try to describe this architecture much more intuitively. Note that in the original paper, the size of the input image is 572x572x3, however, we will use input image of size 128x128x3. Hence the size at various locations will differ from that in the original paper but the core components remain the same.

Below is the detailed explanation of the architecture:
Detailed UNET Architecture
Points to note:

    2@Conv layers means that two consecutive Convolution Layers are applied
    c1, c2, …. c9 are the output tensors of Convolutional Layers
    p1, p2, p3 and p4 are the output tensors of Max Pooling Layers
    u6, u7, u8 and u9 are the output tensors of up-sampling (transposed convolutional) layers
    The left hand side is the contraction path (Encoder) where we apply regular convolutions and max pooling layers.
    In the Encoder, the size of the image gradually reduces while the depth gradually increases. Starting from 128x128x3 to 8x8x256
    This basically means the network learns the “WHAT” information in the image, however it has lost the “WHERE” information
    The right hand side is the expansion path (Decoder) where we apply transposed convolutions along with regular convolutions
    In the decoder, the size of the image gradually increases and the depth gradually decreases. Starting from 8x8x256 to 128x128x1
    Intuitively, the Decoder recovers the “WHERE” information (precise localization) by gradually applying up-sampling
    To get better precise locations, at every step of the decoder we use skip connections by concatenating the output of the transposed convolution layers with the feature maps from the Encoder at the same level:
    u6 = u6 + c4
    u7 = u7 + c3
    u8 = u8 + c2
    u9 = u9 + c1
    After every concatenation we again apply two consecutive regular convolutions so that the model can learn to assemble a more precise output
    This is what gives the architecture a symmetric U-shape, hence the name UNET
    On a high level, we have the following relationship:
    Input (128x128x1) => Encoder =>(8x8x256) => Decoder =>Ouput (128x128x1)

Below is the Keras code to define the above model:
Training

Model is compiled with Adam optimizer and we use binary cross entropy loss function since there are only two classes (salt and no salt).

We use Keras callbacks to implement:

    Learning rate decay if the validation loss does not improve for 5 continues epochs.
    Early stopping if the validation loss does not improve for 10 continues epochs.
    Save the weights only if there is improvement in validation loss.

We use a batch size of 32.

Note that there could be a lot of scope to tune these hyper parameters and further improve the model performance.

The model is trained on P4000 GPU and takes less than 20 mins to train.
9. Inference

Note that for each pixel we get a value between 0 to 1.
0 represents no salt and 1 represents salt.
We take 0.5 as the threshold to decide whether to classify a pixel as 0 or 1.

However deciding threshold is tricky and can be treated as another hyper parameter.
