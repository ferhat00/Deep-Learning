# Deep Learning - Transfer Learning with MobileNet

A practical implementation of transfer learning using Keras and MobileNet for image classification. This project demonstrates how to leverage pre-trained deep learning models and adapt them to classify new categories through transfer learning techniques.

## 📋 Overview

This repository contains a Jupyter notebook that showcases:
- Using pre-trained MobileNet for image classification
- Identifying classification limitations in pre-trained models
- Implementing transfer learning to extend model capabilities
- Training a custom classifier with minimal data

## 🎯 Problem Statement

Pre-trained models like MobileNet are trained on large datasets (e.g., ImageNet with 1000 classes) but may misclassify images that are similar to their training data but belong to different categories. 

**Example from this project:**
- MobileNet correctly classifies various dog breeds (German Shepherd, Labrador, Poodle)
- However, it misclassifies a Blue Tit (European bird) as a Chickadee (North American bird) due to their visual similarity

This project demonstrates how to retrain the model using transfer learning to correctly classify these edge cases.

## 🏗️ Architecture

The project uses **MobileNet**, a lightweight convolutional neural network designed for mobile and embedded vision applications:

- **Size:** ~17MB
- **Efficiency:** Optimized for speed and low computational cost
- **Design:** Uses depthwise separable convolutions
- **Paper:** [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf) (Howard et al., 2017)

## 🔧 How It Works

### 1. Initial Testing with Pre-trained MobileNet
```
Load MobileNet → Test on dog images → Success ✓
                → Test on bird images → Misclassification ✗
```

The notebook first loads MobileNet with ImageNet weights and tests it on various images:
- **German Shepherd** - Correctly classified
- **Labrador** - Correctly classified  
- **Poodle** - Correctly classified
- **Blue Tit** - Incorrectly classified as Chickadee

### 2. Data Collection

Uses `google_images_download` package to automatically download training images:
- 100 images of Blue Tits
- 100 images of Crows
- Filters: JPEG format, size >400x300 pixels

### 3. Transfer Learning Implementation

**Model Architecture Modification:**
```python
Base Model: MobileNet (pre-trained, top layer removed)
    ↓
Global Average Pooling
    ↓
Dense Layer (1024 neurons, ReLU)
    ↓
Dense Layer (1024 neurons, ReLU)
    ↓
Dense Layer (512 neurons, ReLU)
    ↓
Output Layer (2 classes: Blue Tit & Crow, Softmax)
```

**Training Strategy:**
- **Freeze base layers:** Keep MobileNet's pre-trained weights unchanged
- **Train only top layers:** Only the newly added dense layers are trainable
- **Optimizer:** Adam
- **Loss function:** Categorical cross-entropy
- **Batch size:** 32
- **Epochs:** 10
- **Training time:** <2 minutes on GTX 1070 GPU

### 4. Prediction

After training, the model can correctly classify:
- Blue Tits (previously misclassified)
- Crows (new category)

## 📦 Requirements

```python
keras
tensorflow
numpy
IPython
google_images_download
```

## 🚀 Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ferhat00/Deep-Learning.git
   cd Deep-Learning/Transfer\ Learning\ CNN/
   ```

2. **Install dependencies:**
   ```bash
   pip install keras tensorflow numpy google_images_download
   ```

3. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook "Transfer Learning in Keras using MobileNet.ipynb"
   ```

4. **Run the cells sequentially** to:
   - Test pre-trained MobileNet
   - Download training images
   - Build and train the transfer learning model
   - Test predictions on new images

## 📁 Project Structure

```
Deep-Learning/
├── Transfer Learning CNN/
│   ├── Transfer Learning in Keras using MobileNet.ipynb  # Main notebook
│   ├── German_Shepherd.jpg          # Test image
│   ├── labrador1.jpg                # Test image
│   ├── poodle1.jpg                  # Test image
│   ├── blue_tit.jpg                 # Test image (misclassified initially)
│   ├── crow.jpg                     # Test image
│   ├── MobileNet architecture.PNG   # Architecture diagram
│   └── mobilenet_v1.png            # Model structure visualization
└── README.md
```

## 🎓 Key Concepts Demonstrated

1. **Transfer Learning:** Leveraging pre-trained models to solve new problems with minimal data
2. **Feature Extraction:** Using frozen convolutional layers as feature extractors
3. **Fine-tuning:** Adding and training custom classification layers
4. **Data Augmentation:** Using ImageDataGenerator for efficient batch processing
5. **Model Architecture:** Understanding and modifying deep learning architectures

## 🔍 Technical Details

### Image Preprocessing
- Target size: 224x224 pixels (MobileNet input size)
- Preprocessing: MobileNet-specific preprocessing function
- Normalization: Pixel values scaled to [0, 1]

### Model Training
- **Trainable parameters:** Only the last ~20 layers
- **Non-trainable parameters:** Base MobileNet convolutional layers
- **Data flow:** Images → ImageDataGenerator → Batches → Model

## 📊 Results

The transfer learning approach successfully:
- ✅ Maintains accuracy on original ImageNet classes
- ✅ Correctly classifies previously misclassified Blue Tits
- ✅ Adds new classification capability (Crows)
- ✅ Achieves high accuracy with minimal training data (<200 images total)
- ✅ Trains in under 2 minutes on modern GPUs

## 🌟 Extensions

This project can be extended to:
- Add more bird species or animal categories
- Implement data augmentation techniques
- Fine-tune more layers for better accuracy
- Deploy the model as a web service or mobile app
- Use other pre-trained models (ResNet, VGG, EfficientNet)

## 📚 References

- [MobileNets Paper](https://arxiv.org/pdf/1704.04861.pdf)
- [Keras Documentation](https://keras.io/)
- [Transfer Learning Guide](https://keras.io/guides/transfer_learning/)
- [google-images-download](https://github.com/hardikvasa/google-images-download)

## 👤 Author

**Ferhat**

## 📄 License

This project is open source and available for educational purposes.

---

*This project demonstrates the power of transfer learning - adapting powerful pre-trained models to solve specific problems with minimal data and computational resources.*
