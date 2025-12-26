# Intrusion-Detection-System-Using-CNN-and-Transfer-Learning

This is the code for the paper entitled "**[A Transfer Learning and Optimized CNN Based Intrusion Detection System for Internet of Vehicles](https://arxiv.org/pdf/2201.11812.pdf)**" published in **IEEE International Conference on Communications (IEEE ICC)**, doi: [10.1109/ICC45855.2022.9838780](https://ieeexplore.ieee.org/document/9838780).  
- Original Authors: Li Yang and Abdallah Shami  
- Organization: The Optimized Computing and Communications (OC2) Lab, ECE Department, Western University

This repository introduces how to use **convolutional neural networks (CNNs)** and **transfer learning** techniques to develop **intrusion detection systems**.

## 2025 Code Refactoring & Improvements
**Refactored and Reviewed by:** Mohammad Amin Haji Alirezaei
**Email:** **m.a.hajialirezaei05@gmail.com:** This repository has been significantly updated and modernized by **Mohammad Amin Haji Alirezaei** to ensure compatibility with the latest deep learning frameworks. The following improvements have been implemented:

1.  **Migration to TensorFlow 2.x / Keras 3.x:** - The codebase has been fully refactored to remove dependencies on legacy TensorFlow 1.x. All imports now utilize the standard `tensorflow.keras` API.
2.  **Modern Data Pipeline:** - Replaced the deprecated `ImageDataGenerator` with the modern `image_dataset_from_directory` API.
    - Implemented `tf.data` optimizations including `.cache()` and `.prefetch(buffer_size=tf.data.AUTOTUNE)` to significantly speed up training.
    - Integrated `Rescaling` layers directly into the pipeline for efficient normalization.
3.  **Robust Weight Handling:** - Fixed the `HTTP 403 Forbidden` error often encountered when downloading VGG16 weights by implementing a robust local fallback mechanism.
4.  **Shape Mismatch Fixes:** - Resolved conflicts between data generators and loss functions. The code now standardizes on `sparse_categorical_crossentropy` with Integer Labeling mode to prevent shape mismatch errors (e.g., `(None, 1)` vs `(None, 5)`).
5.  **Hyperparameter Optimization Update:** - Updated the `hyperopt` objective functions to be compatible with the new model signatures.

---

- Another **intrusion detection system development code** using **machine learning algorithms** can be found in: [Intrusion-Detection-System-Using-Machine-Learning](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning)

## Abstract of The Paper
Modern vehicles, including autonomous vehicles and connected vehicles, are increasingly connected to the external world. However, this connectivity increases vulnerabilities to cyber-threats. In this paper, a transfer learning and ensemble learning-based IDS is proposed for IoV systems using convolutional neural networks (CNNs). In the experiments, the proposed IDS has demonstrated over 99.25% detection rates on the Car-Hacking dataset and the CICIDS2017 dataset.

<p float="left">
  <img src="https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-CNN-and-Transfer-Learning/blob/main/framework.png" width="500" />
  <img src="https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-CNN-and-Transfer-Learning/blob/main/CAN.png" width="400" /> 
</p>

## Implementation 
### CNN Models  
* VGG16, VGG19, Xception, Inception, Resnet, InceptionResnet

### Ensemble Learning Models
* Bagging, Probability Averaging, Concatenation

### Hyperparameter Optimization Methods
* Random Search (RS)
* Bayesian Optimization - Tree Parzen Estimator(BO-TPE)

### Dataset 
1. **CAN-intrusion/Car-Hacking dataset** (Intra-vehicle intrusion detection)
   * Publicly available at: https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset  
2. **CICIDS2017 dataset** (Network traffic intrusion detection)
   * Publicly available at: https://www.unb.ca/cic/datasets/ids-2017.html  

### Code Structure
* [1-Data_pre-processing_CAN.ipynb](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-CNN-and-Transfer-Learning/blob/main/1-Data_pre-processing_CAN.ipynb): Data transformation (tabular to images).  
* [2-CNN_Model_Development&Hyperparameter Optimization.ipynb](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-CNN-and-Transfer-Learning/blob/main/2-CNN_Model_Development%26Hyperparameter%20Optimization.ipynb): CNN models and hyperparameter optimization (Updated).
* [3-Ensemble_Models-CAN.ipynb](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-CNN-and-Transfer-Learning/blob/main/3-Ensemble_Models-CAN.ipynb): Ensemble learning techniques.

### Libraries (Updated)
* Python 3.8+
* [TensorFlow 2.x](https://www.tensorflow.org/) (Keras API included)
* [OpenCV-python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
* [hyperopt](https://github.com/hyperopt/hyperopt) 
* Pandas, Numpy, Matplotlib, Scikit-learn

## Contact-Info (Original Authors)
* Email: [liyanghart@gmail.com](mailto:liyanghart@gmail.com) or [Abdallah.Shami@uwo.ca](mailto:Abdallah.Shami@uwo.ca)
* GitHub: [LiYangHart](https://github.com/LiYangHart) and [Western OC2 Lab](https://github.com/Western-OC2-Lab/)

## Citation
If you find this repository useful in your research, please cite this article as:  

L. Yang and A. Shami, "A Transfer Learning and Optimized CNN Based Intrusion Detection System for Internet of Vehicles," ICC 2022 - IEEE International Conference on Communications, 2022, pp. 2774-2779, doi: 10.1109/ICC45855.2022.9838780.
