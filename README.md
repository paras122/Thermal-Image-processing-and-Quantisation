Problem Being Solved:

The presence of Non uniformity in thermal imagers results in different type errors that can be seen in the generated scene: striped noise (high frequency), ghosting/motion blur (temporal), lens distortions. 

We are creating a CNN network that takes a greyscale, noisy, thermal image as input and produce a clean output. Essentially we are trying to eliminate the high frequency noise from the images. 

We have broken the problem down into two sections that are quantisation and the Temporal aspect of the model. 

Quantisation:
This part focuses on training the model in a such a way, that after training, the weights produced are such that when quantised to values (-1,0,1) they give the best output. This is being trained on grayscale images that are independent of each other. 


Temporal:
Focuses on training the model on video data using a buffer system to allow the model to focus better on the noise and not interfere with the actual data in the video. 


Sample Inputs: 

Inputs/outputs for Quantised model:

![54](https://github.com/paras122/Thermal-Image-processing-and-Quantisation/assets/82634952/9609c31c-0421-4ac5-a14d-d0276764c064)


![55](https://github.com/paras122/Thermal-Image-processing-and-Quantisation/assets/82634952/34136a4d-5d8b-4dd8-a5d4-b7b9bb068794)


![54](https://github.com/paras122/Thermal-Image-processing-and-Quantisation/assets/82634952/75045c5b-6431-4244-994d-81c17d127c29)


![55](https://github.com/paras122/Thermal-Image-processing-and-Quantisation/assets/82634952/4c6e44b8-05ac-475a-b7e4-61099280c146)

 
Inputs/outputs for Temporal model:
Input:

<img width="391" alt="Screenshot 2024-04-25 at 4 26 20 AM" src="https://github.com/paras122/Thermal-Image-processing-and-Quantisation/assets/82634952/3c0c56c5-0e68-4229-973c-d302f865a191">

Output:

<img width="385" alt="Screenshot 2024-04-25 at 4 26 28 AM" src="https://github.com/paras122/Thermal-Image-processing-and-Quantisation/assets/82634952/d83a899c-5a07-4aa9-944f-c60849d60f13">

Desired Output:

<img width="386" alt="Screenshot 2024-04-25 at 4 26 36 AM" src="https://github.com/paras122/Thermal-Image-processing-and-Quantisation/assets/82634952/d233fe05-9e42-4f5e-9a2e-34a7f3cc8382">


High level diagram of model

Single module/block being used

<img width="419" alt="Screenshot 2024-04-25 at 4 25 52 AM" src="https://github.com/paras122/Thermal-Image-processing-and-Quantisation/assets/82634952/f0da4225-a1bf-4b6d-949e-671e34bc6489">

To be noted that the quantized model is not using video data, thus is not making use of the buffers, whereas the Temporal model is using buffers as shown in the model. 
The Buffers are used in a parallel manner to the convolutional layers. They store the previous feature maps and update the incoming feature maps. 


Intuition behind model (or reference papers):

Quantised Model: 
The quantised model is using a double U-Net, where the kernel size of the convolutional layers decrease till a certain point and then the image is upscaled to desired resolution. The application of varying kernel size is used to provide local and global context of the image, which helps in reconstruction of the downsampled image. The varying kernel size is also being used to identify and differentiate between the high frequency noise in the image, making it easier to subtract the noise from the noisy image. 

Reference: Ma, Shuming, et al. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv preprint arXiv:2402.17764 (2024).


Temporal Model:
The temporal model is using a U-Net architecture with diamond shaped modules for multi scale feature extraction. In each layer, the tensor is subjected to two simultaneous convolutions with different filter sizes and then outputs of the convolutions are added and fed forward into the next layer.
In addition, buffers are being used along with each layer to save the output and reuse it to better denoise the next incoming image. This provides the model with a temporal context to the incoming data and should help it better differentiate between features that are actually in the image and the noise.


Current Performance: 

Quantised Model

<img width="926" alt="Screenshot 2024-04-25 at 4 23 57 AM" src="https://github.com/paras122/Thermal-Image-processing-and-Quantisation/assets/82634952/4f5231b4-0ed0-4607-bb0e-310b492d5da4">

Validation Loss (Red), Training Loss (Blue)


Weights of 6th convolutional layer vs Epochs ( while training)

<img width="727" alt="Screenshot 2024-04-25 at 4 24 57 AM" src="https://github.com/paras122/Thermal-Image-processing-and-Quantisation/assets/82634952/b6a6e8b0-ed9a-4de6-97c4-eadb1c4830e2">

Inference from graphs:

Quantised Model

The validation from the first graph shows us how the training loss and error loss are matching up. It is clearly visible that the training loss is very small from the beginning and does not learn much. This can be attributed to the error function, as it is providing a very low error but the real performance is not good. This explains why the model is performing badly on the validation set. The error function needs to be updated in such a manner that it correctly directs the model to learn the desired traits. 

The graph of the weights of the 6th convolutional layer is indicative of probable vanishing gradient issue. This is because the weights quickly converge to zero, only after a few epochs. This is restricting and learning and also explains the output received from the 6th layer. 


Temporal Model:
Currently, we have not been able to make the buffer system work. Problems are arising during gradient computation due to the tensor being updated in place. If we decouple the buffer system from the gradient descent step, results are not very good as the denoised output gets blurry and distorted. 

Also, while training a classic U net model without buffers using SSIM or MS_SSIM loss is 0 right from the start indicating that the output is perfect. Loss isn't changing even after multiple epochs. Using L1 or MSE loss, output image is getting denoised but it is also becoming blurry.

NOTE: This is the model update until now. Ongoing work is being done in perfecting the model. The 1-bit math has still not been applied in terms of memory usage and operations but the values are quantised to (-1,0,1) in the quantised model. 
