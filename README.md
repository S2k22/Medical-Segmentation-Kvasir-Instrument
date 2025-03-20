# Introduction
This work is a part of experimental study. In this work I tried different Deep Learning models to solve a medical segmentation problem. The data set I will perform my experiments on contains endoscopic images of tools and their ground truth masks.
> [!IMPORTANT]  
> Unfortunately Github doesn't allow previews for this Jupyter Notebook because it's too large. Because of that I attached a pdf file of my code so people who are interested still can explore it from github. But the code is also available for downloading. I will also discuss the result and choice in this paper.

## The dataset
The Kvasir-Instrument dataset (size 170 MB) contains 590 endoscopic tool images and their ground truth mask. The resolution of the image in the dataset varies from 720x576 to 1280x1024. The image file is encoded using jpeg compression. The dataset comes with pre-made test and train splits and in this experiment I will use it as it is. But I also left the option to create a custom split in the code.
### The link to a dataset
https://datasets.simula.no/kvasir-instrument/

## Metrics
In this experiment I used a standard metrics such as:
- Train loss/Validation Loss
- Validation Accuracy/Train Accuracy
- Train Dice/Validation Dice
- Train IoU/Validation IoU
- Best Epoch

![image](https://github.com/user-attachments/assets/c75e27db-7fda-4e2f-b15e-daaf0ee22bc6)

![image](https://github.com/user-attachments/assets/38133cf3-7a4d-4915-a685-48d8d0da97e3)

![image](https://github.com/user-attachments/assets/7d0e9d26-33fe-427e-a323-d55eb5bf7448)



I have also used a separate function to evaluate the model using the same metrics but on Test Split. The process of this goes:
- Saving the best model during training (weights) according to the best validation loss,
- load the best model,
- setting it to evaluation mode,
- computing metrics,
- averaging across the batches,
- storing metrics.
 
This pipeline ensures thattest metrics reflect the performance of the best model on unseen test data, without influencing training decisions.

## Activation Function (sigmoid)
I use the BCEloss function in my work.  It transforms the raw network outputs (logits) into values between 0 and 1, which can be interpreted as probabilities for the binary classes (object vs. background).
By applying the sigmoid activation, the model’s outputs meet this requirement, ensuring that the loss is computed correctly.
The output values after the sigmoid make it easier to apply a threshold to convert the continuous probability map into a binary segmentation mask.

![image](https://github.com/user-attachments/assets/c9be24a4-d5fe-4aab-971e-36eee91b8785)

## Data preparation
First step is setting dataset directories and splits.

Next we create a custom PyTorch Dataset class that takes in a list of image paths and mask paths. It reads images and converts the BGR image to RGB. The mask is read in grayscale, and then thresholded to obtain a binary mask.



Then we apply the default transformation pipeline. This default pipeline resizes the image and mask to 256×256, normalizes the pixel values, and converts them to tensors. And then we make sure that the mask has the expected shape.

After that we do data splitting and verification (making sure that the data set correctly before training)

## Augmentations
I used the Albumentations library to define augmentations in this work.
- A.HorizontalFlip(p=0.5). With 50% chance flips the image horizontally.
- A.VerticalFlip(p=0.5). Similarly, with a 50% probability, this flips the image vertically.
- A.RandomRotate90(p=0.5). This rotates the image by 90 degrees, 180 degrees, or 270 degrees randomly with a 50% chance.
- A.RandomBrightnessContrast(p=0.2). With a 20% chance, this randomly adjusts the brightness and contrast of the image.
- A.GaussNoise(p=0.2). This adds Gaussian noise to the image with a 20% probability.
- A.Resize(256, 256). This resizes both the image and the mask to 256×256 pixels.
- A.Normalize. This normalizes the pixel values based on the default mean and standard deviation
- ToTensorV2. Converts the processed images and masks into PyTorch tensors.

There is also a separate option called No Augmentation, it only consist of resize, normalize and converting to tensor, it is made so I could train model with or without augmentation.

# Models
In this experiment I have used and compared 3 Deep Learning models: U-Net, U-Net++, FPN. And in this section I will try to explain how they work and their structure.

## U-net
My U-Net model follows a classic encoder–decoder network designed for image segmentation. The design emphasizes a symmetric architecture with skip connections that transfer high-resolution features from the encoder to the decoder.

Encoder:
- Convolutional Blocks. Each block applies two successive 3×3 convolutions, each followed by Batch Normalization and a ReLU activation.
- Downsampling. After each block, a max pooling layer reduces the spatial dimensions by a factor of 2, which helps the network learn context and global features.

Bottleneck:
- Located at the last parts of the network, the bottleneck is made to capture the most abstract and compressed representation of the input. It is implemented with the same convolutional block structure, but with a higher number of features (channels).

Decoder:
- Upsampling. The decoder uses transposed convolutions to upsample feature maps. This increases the spatial dimensions gradually.
- Skip Connections. At each decoding level, the upsampled feature map is concatenated with the corresponding encoder output. These skip connections help recover spatial details lost during downsampling.
- Decoder Blocks. The concatenated features are then processed with additional convolutional blocks to refine the segmentation map.

## U-net++
UNet++ builds on the original UNet design by introducing nested skip connections and feature re-aggregation. The idea is to bridge the gap between decoder and encoder and improve performance.

Encoder:
- Pretrained Backbone. The model uses a pretrained ResNet34 as its encoder. This backbone provides robust, multi-scale features and consists of 4 layers starting with 64 and doubling its size till it reaches 512 in the last layer.
- The layers are processed sequentially, with intermediate downsampling similar to UNet.

Decoder:
- Structure. The custom UNetPlusPlusDecoder accepts encoder outputs in the order [x4, x3, x2, x1] (from deepest to shallowest) and processes them using nested convolution blocks.
- Convolution Blocks. The decoder creates intermediate outputs (e.g., x0_0, x1_0, x1_1, etc.) by progressively upsampling deeper features and concatenating them with features from shallower layers.
- Upsampling. Features are upsampled using interpolation to match the spatial dimensions before concatenation.
- Refinement. After concatenation, convolutional blocks refine the merged features.

Final Segmentation:
- A segmentation head (a 3×3 convolution) produces the output feature map, which is then upsampled to match the original input size.
- The final activation is a sigmoid to output probabilities for binary segmentation.

## FPN
Designed to leverage features at multiple scales, making it effective for detecting objects of various sizes.
It builds a pyramid of features by combining high-level semantic information with low-level spatial details.
Encoder:
- Supports the same pretrained ResNet34 or ResNet50. The difference between these two is in the number of channels. ResNet50 starts the first layer with 256 channels.
- The encoder first processes the image through an initial convolution, batch normalization, ReLU, and max pooling before extracting features from subsequent layers.

Decoder:
- Pyramid Construction. The decoder builds multiple feature maps, starting from the deepest encoder feature
- FPNBlock. Each FPNBlock takes a higher-level feature, upsamples it (using nearest or bilinear interpolation), and adds it to a transformed version of a skip connection from the encoder. This is typically performed after aligning spatial dimensions.
- Segmentation Blocks. Each pyramid level is further processed by a segmentation block that applies a series of 3×3 convolutions with Group Normalization and ReLU. These blocks ensure that all pyramid features reach a common resolution.

Feature Merging:
- The refined features from each pyramid level are merged together using a merging block. The merge policy in my code is set to "add", meaning that features are combined via element-wise summation.

Final Segmentation:
- A final segmentation head reduces the merged feature map to the desired number of output channels. The result is then upsampled to match the original input dimensions, and a sigmoid activation produces the final segmentation map.

## Optimizer, Learning Rate Scheduler and Hyperparametrs.

Adam Optimizer:
- Adam combines the benefits of both Momentum and RMSprop. It maintains per-parameter learning rates and adapts them based on estimates of first and second moments of the gradients.

Reduce on Plateau:
- This scheduler monitors a specific metric (in my case, the validation loss) and reduces the learning rate by a given factor (0.5) if the metric does not improve for a specified number of epochs (patience=2 in my case).

Hyperparameters:
- I didn't include too many parameters in this experiment but the ones that are there were chosen after a couple of experiments (not all of them were included). It is worth mentioning that Adam often gets used with a small learning rate (0.001 or even 0.0001) in my experiment I started with 0.1 and decreased it in the next experiments.

## Running models, Results and Explanation.

The number of epochs was set to a 60, not all the models (across different experiments) reached to the 60' epoch, the early stopping prevented the model to go further with no development after patience reached a value of 5. The three experiments that I want to highlight in this paper was (The upper model is FPN, the middle is U-Net, the down one is U-Net++) and:

### Models + No Augmentation and Learning Rate of 0.1

![image](https://github.com/user-attachments/assets/d2733556-3860-45a7-ad0a-d523d6a0765c)


### Models + Augmentation and Learning Rate of 0.01

![image](https://github.com/user-attachments/assets/b990030e-82c6-4016-b5a3-75031f3bc574)


### Models + Augmentation and Learning Rate of 0.001

![image](https://github.com/user-attachments/assets/60359995-1114-4a38-a117-0f7f205af3d8)


Not surprisingly but model with augmentation performed better than the one without it, with one exception. And across all 3 experiments models with Augmentation + a learning rate of 0.001 performed the best. The best model, out of the 3 presented, according to a test metrics were U-Net and the second best was FPN.
It is worth mentioning that in the first experiment with no augmentation and relatively high learning rate for this optimizer, U-net ++ outperformed the rest of the models.
If I have to make a suggestion why that happened then I would probably say because augmentation and high learning rate did not allow other models to get available variability and take advantage of high dimensionality, also more complex models(FPN) tend to overfit with no augmentation. Under this conditions nested skip connections might help the model explore a larger space of representations quickly
and the aggressive learning rate forces rapid changes. But when the augmentation was applied and learning rates were tuned down it caused slower convergence and maybe an overfitted model more than before. In other words its complexity is a double-edged sword.

## I will highlight the best models out of each experiment in this section

### Predictions U-Net + No Augmentation and Learning Rate of 0.1

![image](https://github.com/user-attachments/assets/7a891122-06c5-4e50-b417-43ed5c4d8d5e)

### Predictions U-Net++ + No Augmentation and Learning Rate of 0.01

![image](https://github.com/user-attachments/assets/01c2eca7-7c4b-442a-8e8c-0799e58ae926)

### Predictions U-Net++ + No Augmentation and Learning Rate of 0.01

![image](https://github.com/user-attachments/assets/2f35aa8c-ddff-4f17-ad3e-bb42c916206b)


## Conclusion
In conclusion we can say that the classic-cal U-Net architecture outperforms the UNet ++ model. Additionally, U-Net is 2× faster than the UNet++. This is because U-Net uses basic convolution blocks, whereas DoubleUNet uses a pre-trained encoder. More fine-tuning and adjusting the hyperparameters on the similar datasets, adjusting the augmentation can improve the results I observed.
With the goal of perfection the performance on the localization and segmentation will theoretically allow to make this technology useful in the medical environment.
 







  
