# Imago

An Image Segmentation AI Tool, which leverages u-net and u2-net architectures to achieve single and multi-segmentation, and object-cropping in images.

### Inspiration

Image segmentation was always so cool to me, especially the visualizations that were created to crop images and segment them. But I never thought it would be so complex...

Now here I am dabbling with the ML structure directly to understand object detection and cropping.

### Topics

- Languages: Python
- Architectures: U-NETs, U2-NETs, Single/Multi-segmentation, Object-Cropping

### Use It Yourself

It is as simple as cloning, installing the right python dependencies as prompted, and running the Python file in any IDE with the right interpreter.

The Juypter Notebooks follow the same procedure above, just run the entire file or any cell given the right IDE, environment, and dependencies.

### Architectures (In Detail)

It is for those who are are curious/interested, in the underlying architecture of each of the model architectures used in Imago.

#### U-NET

U-Net is a popular convolutional neural network architecture used primarily for image segmentation tasks in the field of computer vision.

The architecture of a U-Net can be divided into two main parts: the "encoder" and the "decoder." Here's an overview of how U-Nets work and their key architectural components:

- Encoder:

  - Responsible for capturing features from the input image. Consists of a series of convolutional layers, typically organized in a downsampling fashion. These layers reduce the spatial resolution of the input image while increasing the number of feature channels.
  - Max-pooling or similar downsampling techniques are often used to reduce the size of the feature maps and capture increasingly abstract features as we move deeper into the encoder.

- Bottleneck:

  - After several downsampling steps, the encoder reaches a bottleneck layer. This layer contains the most abstract and high-level features extracted from the input image.

- Decoder:

  - Responsible for reconstructing the segmented output from the abstract features captured by the encoder. Consists of a series of upsampling and deconvolutional layers.
  - The upsampling layers increase the spatial resolution of the feature maps while reducing the number of feature channels. Skip connections, which are connections from the encoder's layers to the decoder's layers, are a key component of U-Net architecture. These connections help in preserving fine-grained spatial information and improve the quality of segmentation masks.
  - The decoder gradually refines the features and generates a segmentation map that is the same size as the input image.

- Output:

  - Output layer uses a sigmoid activation function, which produces pixel-wise predictions. Output is a binary mask where each pixel in the mask corresponds to the probability of the corresponding pixel in the input image belonging to the object of interest.

#### U2-NET

U2-Net is a deep learning architecture designed for image segmentation tasks, particularly focused on achieving high-quality and precise object segmentation.

The U2-Net architecture is an improved version of the original U-Net architecture and has been designed to address some of its limitations.

U2-Net Architecture:

- Encoder-Decoder Structure:

  - U2-Net, like the original U-Net, employs an encoder-decoder architecture. The encoder extracts high-level features from the input image, while the decoder upsamples these features to produce a segmentation mask.

- Backbone Network:

  - In U2-Net, a key improvement is the use of a more powerful backbone network. It employs a VGG-like structure with a modified ResNet architecture as its backbone. This helps the model capture complex and multi-scale features from the input image.

- Dilated Convolutions:

  - U2-Net utilizes dilated convolutions in its encoding and decoding layers. Dilated convolutions allow for an increased receptive field without increasing the number of parameters, which is important for capturing details and context in segmentation tasks.

- Attention Gates:

  - One of the unique features of U2-Net is the incorporation of attention gates. Attention mechanisms help the model focus on relevant regions while ignoring irrelevant information, improving the segmentation accuracy.

- Nested Feature Pyramid:

  - U2-Net incorporates a nested feature pyramid structure, which enables the network to capture features at multiple scales. This helps in handling objects of various sizes in the image.

Implementation Structure in Imago {Diagrams}:

U Block Comparisons:
<img src="https://github.com/ReshiAdavan/Imago/blob/master/imgs/U-Block-RSU.PNG" />

U2-NET Proposed Architecture:
<img src="https://github.com/ReshiAdavan/Imago/blob/master/imgs/U2-NET-Architecture.PNG" />

#### Comparisons

U2-NET is an extension of the U-NET architecture with a deeper and more complex design, making it more suitable for tasks that require precise and detailed image segmentation. U-NETs are more popular but may not perform as well as U2-NET in scenarios with highly intricate object boundaries and structures.

If you made it this far, congrats! That concludes Imago's README.
