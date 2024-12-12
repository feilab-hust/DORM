# DORM

This repository contains the source code and instructions for **DORM**. You should be able to download the code and set up the environment.

## Requirements

DORM is developed using MATLAB, Python, and PyTorch. The code is optimized for Windows and has been thoroughly tested in this environment.

The inference process has been validated on the following configuration:

- Windows 10
- MATLAB 2022a
- Python 3.9.12 (64-bit)
- PyTorch 1.11.0
- Intel Core i9-10900X CPU @ 3.70GHz
- Nvidia GeForce RTX 3090

## Installation
   **1. MATLAB 2022a**

   Please refer to the official MathWorks documentation for installation instructions for MATLAB 2022a. You can find it on the MathWorks website under the "Installation Guide" section.

   **2.Depp learning environment**
   1. Create a new environment based on Python 3.9.12:
    ```bash
    conda create -n DORM python=3.9.12
    ```
   2. (Optional) If your computer has a CUDA-enabled GPU, install the appropriate versions of CUDA and cuDNN.
   3. Install PyTorch:
       ```bash
       conda install pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 -c pytorch
       ```
   4. Download the `DORM.zip` file and unpack it.
   5. Open a terminal in the `DORM` directory and install the dependencies using pip:
       ```bash
       pip install -r requirements.txt
       ```


## Training Process

1. Open a terminal in the `DORM/code` directory and run `train.py`; the 'Config train' panel will appear.

2. Set the parameters:

    - **Label tag:** Add a specific tag to the currently trained network.
    - **GT path:** Click the ‘Choose’ button and select the folder containing GT(ground truth) data on your computer.
    - **Patch size:** Recommended values: Depth, Height, Width = 32, 64, 64.
    - **Threshold:** Default value is 0.9. For most data, this is sufficient.
    - **Gaussian noise:** Set the standard deviation of Gaussian noise added to Raw data. Default value is 0, meaning no Gaussian noise.
    - **Poisson noise:** Check this option to add Poisson noise to Raw data; uncheck to omit it.
    - **Raw data path:** Click the ‘Choose’ button and select the folder containing Raw data on your computer.
    - **GAN model:** Choose whether to use the GAN model during training.

<div align="center">
<img width="480" height="240" src="/fig/Config_train.png"/>
</div>

3Click the ‘Start running’ button to begin the training process.

## Inference Process

Due to data size limitations, the pre-trained models are stored [here](https://drive.google.com/file/d/1uLWXmoXxNQB0pUhR1EgzYKN73NxkLBjF/view?usp=sharing). You will need to download and unzip them into the `experiments` folder of the original DORM demo. Example data for testing the models on various tasks can be downloaded [here](https://drive.google.com/file/d/1FBkRGqa5LsvJEwLW29WGEEcaKYS_Qbzi/view?usp=sharing).


1. Open a terminal in the `DORM/code` directory and run `inference.py`; the 'Config inference' panel will appear.

2. Set the parameters for inference:

    - **Label tag:** The label tag of the converged neural network you want to use.
    - **Validation path:** Click the ‘Choose’ button to select the validation data you have downloaded.

<div align="center">
<img width="480" height="180" src="/fig/Config_inference.png"/>
</div>

3. Click the ‘Start running’ button to begin inference. The model outputs will be saved in a newly created subfolder within the “Validation path.”

## Acknowledgements

This program was developed using deep learning via PyTorch. We also acknowledge the generous contributions of Xintao Wang et al.[1] and Martin Weigert et al.[2]. You are welcome to use the code or program freely for research purposes. For further inquiries, please contact us at feipeng@hust.edu.cn or chenlongbiao@hust.edu.cn.

## References

[1] Xintao Wang, Liangbin Xie, Ke Yu, Kelvin C.K. Chan, Chen Change Loy, and Chao Dong. BasicSR: Open Source Image and Video Restoration Toolbox. https://github.com/xinntao/BasicSR, 2022.

[2] Weigert, M., Schmidt, U., Boothe, T., et al. Content-aware image restoration: pushing the limits of fluorescence microscopy. Nat Methods 15, 1090–1097 (2018).
