Here’s a template for your README file, based on your project description and decisions:

---

# Sentiment Analysis Using Deep Learning and Machine Learning Models

This project focuses on sentiment analysis of tweets using both deep learning models (RNN, LSTM, GRU) and a machine learning model (Random Forest). We explore how these models perform on a dataset of tweets, aiming to predict sentiment polarity (positive, negative, or neutral). Despite applying regularization techniques and hyperparameter tuning, the results show signs of overfitting, pointing to the need for better data preprocessing and quality improvement.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Dependency Installation](#dependency-installation)
- [Results](#results)

## Project Overview

The main objective of this project is to predict the sentiment of tweets using four models:

1. **Random Forest** (Machine Learning)
2. **RNN** (Deep Learning)
3. **LSTM** (Deep Learning)
4. **GRU** (Deep Learning)

We tried various regularization techniques, including dropout and hyperparameter tuning, to address overfitting. Despite these efforts, overfitting remained a challenge, likely due to limitations in the dataset quality.

## Dataset

The dataset used for this project consists of tweets labeled with sentiment. It contains various text features, and the tweet texts were preprocessed to clean and convert them into numerical vectors. The dataset was split into training, validation, and test sets.

### Preprocessing Steps:
- Removal of null values and noise
- Tokenization and stop-word removal
- Word2Vec embedding to convert text tokens into vectors
- Padding for handling variable-length inputs

## Models Used

### Random Forest
- Used for baseline sentiment classification
- Hyperparameters: `n_estimators=10`, `random_state=50`

### RNN, LSTM, and GRU
- Deep learning models to capture sequential patterns in tweet texts
- Configuration for all models:
  - Input Size: 300 (word embedding dimension)
  - Hidden Size: 64 units
  - Hidden Layers: 1 layers
  - Optimizer: Adam with a learning rate of 0.0001
  - Loss Function: Cross-entropy loss
  - Epochs: 30
  - Dropout: 0.1 to avoid overfitting

## dependency-installation

To run this project, follow these steps:

1. Instructions to Install Visual Studio 2022 Community Edition

PyTorch relies heavily on C++ extensions for performance optimization. One of the fastest ways to ensure these dependencies are properly set up is to install **Visual Studio 2022 Community Edition**. Follow these steps to complete the installation:

**Download Visual Studio 2022 Community Edition**:
   - Use this [Visual Studio 2022 Community Edition](https://visualstudio.microsoft.com/vs/community/) to download the installer.

**Install the Visual Studio Installer**:
   - Run the `.exe` file that you downloaded.

**Install the Required C++ Components**:
   - During the installation, ensure you select the components for **Desktop development with C++**.
   
**Complete the Installation**:
   - Let the Visual Studio installer download and install the necessary files. This will set up the C++ toolchain, which PyTorch depends on.

Once this process is complete, your system will have the necessary C++ extensions to run PyTorch properly.


2. Instruction for install dependent Libraries

   ```bash
   -m pip install -r requirements.txt
   ```

3. Instruction to download GoogleNews word2vec pretrain weights

Kindly go through the following steps-

1. Download the [googlenews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?resourcekey=0-wjGZdNAUop6WykTtMip30g)
2. Copy the file and paste it in the current working directory.



## Results

Despite applying various regularization techniques and hyperparameter tuning, all models showed overfitting tendencies. This suggests that the primary issue lies with the dataset quality. 

### Key Findings:
- The **GRU model** performed slightly better in terms of training accuracy, but overfitting was observed in all models.
- **Random Forest**, while faster to train, also struggled with generalization on unseen data.

### Performance Metrics:
- Training, validation, and test accuracies for each model are available in the results folder.
