# Applied AI Midterm Exam: Super Resolution GAN (SRGAN) Implementation

This project is part of the Midterm Exam for Applied AI. It aims to implement a Super Resolution Generative Adversarial Network (SRGAN) to enhance low-resolution images and subsequently use them in a binary classification problem for cat and dog images. The final model's performance is compared with a baseline model.

## Introduction: Super Resolution GAN (SRGAN)

Super Resolution GAN (SRGAN) is a deep learning architecture that combines GANs and Convolutional Neural Networks (CNNs) to generate high-resolution images from low-resolution inputs. The SRGAN architecture consists of a generator that attempts to produce high-resolution images and a discriminator that tries to distinguish between real and generated high-resolution images. The training process involves using the feedback from the discriminator to iteratively improve the generator's ability to produce realistic high-resolution outputs.

SRGAN has applications in medical imaging, satellite imagery, video processing, and other areas where enhancing low-resolution images is beneficial for analysis and decision-making.

## Steps Overview

The project followed these major steps:

1. **Binary Classification Model A using Transfer Learning**: The dataset was downscaled to 128x128, and a binary classifier (Model A) was trained on it using VGG16.
2. **Training SRGAN**: A SRGAN model was trained to upscale 32x32 images to 128x128 high-resolution images.
3. **Training Binary Classification Model B using Generated Images**: The images generated by SRGAN were used to train another classifier (Model B).
4. **Comparison of Models A and B**: Both models were compared using various metrics such as Accuracy, F1 Score, and AUC.

## Data Preparation

- The dataset was split into **70% training** and **30% validation**.
- Images were downscaled to **128x128** for classification purposes and further to **32x32** for SRGAN training.
- **Image augmentations** such as horizontal flipping, rotation, and color jittering were applied to improve model generalization.

## Training Model A: Binary Classifier

Model A was trained on the downscaled 128x128 images. The architecture used is a pre-trained **VGG16** model with the last layer modified to classify between cats and dogs.

- **Transformations Applied**: Resize to 128x128, random horizontal flip, rotation within 15 degrees, color jitter, and normalization.
- **Training and Validation Loss**:

![Training and Validation Loss](./figures/training_validation_loss_A.png)

- **Validation Metrics**: 

![Validation Metrics](./figures/validation_metrics_A.png)

- **Validation Predictions**: 

![Validation Predictions](./figures/validation_predictions_A.png)

- **Confusion Matrix**: 

![Confusion Matrix A](./figures/confusion_matrix_A.png)

## SRGAN Training

The SRGAN model was trained for **150 epochs** to upscale low-resolution (32x32) images to high-resolution (128x128) images.

- **Low-Resolution vs High-Resolution Examples**: 
![Original and Downscaled Images](./figures/srgan_train_data_original_and_downscaled_images.png)
- **Validation Sample Results**: 
![Validation Samples](./figures/srgan_validation_sample_images.png)
- **Generator and Discriminator Loss Plot**: 
![SRGAN Loss Plot](./figures/srgan_loss_plot.png)

## Training Model B: Using Generated Images

Model B was trained using the high-resolution images generated by SRGAN. These images, along with original training images, were used to create a larger dataset for training.

- **Transformations Applied**: Similar transformations were applied as in Model A.
- **Training and Validation Loss**:

![Training and Validation Loss](./figures/training_validation_loss_B.png)

- **Validation Metrics**: 

![Validation Metrics](./figures/validation_metrics_B.png)

- **Validation Predictions**:

![Validation Predictions](./figures/validation_predictions_B.png)

- **Confusion Matrix**: 

![Confusion Matrix B](./figures/confusion_matrix_B.png)

## Model Comparison

The performance of both models, A and B, was compared on the validation dataset:

### Metrics for Model A
- **Validation Loss**: 0.2068
- **Accuracy**: 95.27%
- **F1 Score**: 95.27%
- **AUC**: 95.27%

### Metrics for Model B
- **Validation Loss**: 0.1361
- **Accuracy**: 95.68%
- **F1 Score**: 95.68%
- **AUC**: 95.68%

### Observations
- Model B, trained with SRGAN-generated images, achieved slightly better performance compared to Model A, indicating that the generated high-resolution images contributed positively to classification performance.
- Although the differences between Model A and B are not very large, Model B does show relatively better results. This demonstrates that incorporating SRGAN-generated images is effective in improving the classifier's ability to distinguish between classes.

## Conclusion

The SRGAN-generated images improved the binary classification performance. This demonstrates the potential of using GAN-generated high-resolution images to enhance model training, especially in cases where high-quality images are not readily available.

The complete implementation and results can be reproduced using the provided code and documentation. The training scripts, validation scripts, and all required configuration files are available in this repository.

