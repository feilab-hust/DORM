# Deep-DORM

This repository contains the source code and instructions for **DORM**. You should be able to download the code and set up the environment within five minutes.

## Requirements

Deep-DORM is built with Python and PyTorch. Although the code can run on any operating system, it is recommended to use Windows, which has been tested.

The inference process has been tested with:

- Windows 10
- Python 3.9.12 (64-bit)
- PyTorch 1.11.0
- Intel Core i9-10900X CPU @ 3.70GHz
- Nvidia GeForce RTX 3090

## Installation

1. Create a new environment based on Python 3.9.12:
    ```bash
    conda create -n DORM python=3.9.12
    ```
2. (Optional) If your computer has a CUDA-enabled GPU, install the appropriate versions of CUDA and cuDNN.
3. Install PyTorch:
    ```bash
    conda install pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 -c pytorch
    ```
4. Download the `Deep-DORM.zip` file and unpack it.
5. Open a terminal in the `Deep-DORM` directory and install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

The installation takes about 5 minutes on the tested platform, though it may take longer depending on network conditions.

## Training Process

1. Open a terminal in the `Deep-DORM` directory and run `train.py`; the 'Config train' panel will appear.

2. Set the parameters in **Global parameters**:

    - **Label tag:** Add a specific tag to the currently trained network.
    - **HR path:** Click the ‘Choose’ button and select the raw HR data on your computer. (Full forms of abbreviations are listed in the ‘Abbreviation Table’ of the NOTE at the end. This applies throughout.)
    - **Patch size:** Set ‘Depth’ larger than 1 for SR/denoise nets. Recommended values: Depth, Height, Width = 32, 32, 32. Set ‘Depth’ equal to 1 for ISO net; recommended values: Depth, Height, Width = 1, 32, 32.
    - **Threshold:** Default value is 0.9. For most data, this is sufficient. However, you may adjust the 'Threshold' based on the preview of the generated training data pairs, ensuring minimal non-signal areas and more than 2K training pairs.
    - **Gaussian noise:** Set the standard deviation of Gaussian noise added to LR data. Default value is 0, meaning no Gaussian noise.
    - **Poisson noise:** Check this option to add Poisson noise to LR data; uncheck to omit it.

3. Set the parameters in **3D data parameters** or **ISO parameters**:

    - **For enhancement/SR network:** Configure parameters in ‘3D data parameters’ to generate processed LR data.
        - **LR path:** Click the ‘Choose’ button and select raw LR data on your computer.
        - **GAN model:** Choose whether to use the GAN model during training.

    - **For ISO net:** Configure parameters in the ‘ISO parameters’ panel to generate synthetic anisotropic data.
        - **PSF file:** Add optical blurring to the generated anisotropic x-y slice. First, prepare a 2D Gaussian function (.txt file) simulating the axial PSF of your 3D datasets. Then, click ‘Choose’ to load the PSF file from your computer. The program will generate final synthetic x-y slices with axial resolution similar to the z-y slices. These degraded x-y slices will be paired with raw x-y slices for model training.
        - **Axis subsample:** Resample the isotropic x-y slices of the 3D datasets into anisotropic ones. For example, setting this value to 2 will down-sample the x-y slices by a factor of 2 along the x direction, generating resampled x-y slices that simulate the anisotropic z-y slices of the 3D datasets.

**NOTE:** When generating data for ISO-net, which processes a 3D image stack slice by slice, set the depth of patch size to 1. In this case, the ‘3D data parameters’ panel will be deactivated, and only the parameters in the ‘Global parameters’ and ‘ISO parameters’ panels will be available. Conversely, when generating data for the enhancement or SR network, where a 3D image stack is treated as a whole volume, set the depth of patch size greater than 1. In this case, the ‘ISO parameters’ panel will be deactivated, and datasets for the enhancement or SR network will be generated.

<div align="center">
<img width="480" height="360" src="/fig/Config_train.png"/>
</div>

4. Click the ‘Start running’ button to begin the training process.

## Inference Process

Due to data size limitations, the pre-trained models are stored [here](https://drive.google.com/file/d/15irFFxQ09njuqFAXpkY5o3_TUvbqMj_6/view?usp=sharing). You will need to download and unzip them into the `experiments` folder of the original Deep-DORM demo. Example data of various organelles for testing the model can be downloaded [here](https://drive.google.com/file/d/15l1izMYDFhMMOf0AXF_p9py6hhy04PTX/view?usp=sharing).


1. Open a terminal in the `Deep-DORM` directory and run `inference.py`; the 'Validation' panel will appear.

2. Set the parameters for inference:

    - **Label tag:** The label tag of the converged neural network you want to use.
    - **Net type:** Use "ISO_net" for ISO enhancement, and "3D_net" for all other cases.
    - **Validation path:** Click the ‘Choose’ button to select the validation data you have downloaded.

<div align="center">
<img width="480" height="180" src="/fig/Config_inference.png"/>
</div>

3. Click the ‘Start running’ button to begin inference. The model outputs will be saved in a newly created subfolder within the “Validation path.”

## Acknowledgements

This program was developed using deep learning via PyTorch. We also acknowledge the generous contributions of Xintao Wang et al.[1] and Martin Weigert et al.[2]. You are welcome to use the code or program freely for research purposes. If you publish work that benefits from this program, please cite it as "XXX". For further inquiries, please contact us at feipeng@hust.edu.cn or chenlongbiao@hust.edu.cn.

## References

[1] Xintao Wang, Liangbin Xie, Ke Yu, Kelvin C.K. Chan, Chen Change Loy, and Chao Dong. BasicSR: Open Source Image and Video Restoration Toolbox. https://github.com/xinntao/BasicSR, 2022.

[2] Weigert, M., Schmidt, U., Boothe, T., et al. Content-aware image restoration: pushing the limits of fluorescence microscopy. Nat Methods 15, 1090–1097 (2018).
