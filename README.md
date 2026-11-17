# Flower Classification with Transfer Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/varad-takale/flower-classification-transfer-learning/blob/main/notebooks/transfer_learning_flowers102.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Comparing three pre-trained CNNs (**ResNet50**, **VGG16**, **MobileNetV2**) on the **Oxford Flowers 102** dataset using transfer learning with frozen ImageNet backbones.

## Results

Test-set accuracy on 6,149 unseen images (102 classes):

| Model | Test accuracy |
|---|---|
| **ResNet50** | **83.05%** |
| MobileNetV2 | 76.87% |
| VGG16 | 54.69% |

ResNet50 was the best model, MobileNetV2 was a lighter alternative with lower accuracy, and VGG16 was the weakest.

### Confusion matrix (ResNet50)

![Confusion matrix](images/confusion_matrix_resnet50.png)

The notebook also prints a per-class precision/recall report. Performance varies a lot by class: some classes are predicted almost perfectly, while others are weak, so overall accuracy hides large per-class differences.

### Sample predictions (ResNet50)

![Sample predictions](images/sample_predictions.png)

*Nine test images from one batch. All nine happen to be correct here, so this is an illustration and not a measure of accuracy.*

## Dataset

[Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), loaded through TensorFlow Datasets (`oxford_flowers102`).

- 8,189 images in 102 flower categories
- Split used: **1,020 train / 1,020 validation / 6,149 test** (about 10 training images per class)

## Approach

1. Load the dataset with `tensorflow_datasets` (train / validation / test splits).
2. Preprocess: resize to 224x224, one-hot encode labels, shuffle, batch (32) and prefetch. Each model applies its own `preprocess_input` inside the model.
3. Build each model: pre-trained ImageNet base (top removed, **frozen**) + `GlobalAveragePooling2D` + `Dense(102, softmax)`.
4. Train with Adam and categorical cross-entropy for up to 10 epochs, with `EarlyStopping` (patience 3, restore best weights) and `ModelCheckpoint`.
5. Evaluate on the test set, then analyze ResNet50 with a confusion matrix, a classification report and sample predictions.

## Tech stack

Python, TensorFlow / Keras, TensorFlow Datasets, NumPy, scikit-learn, Matplotlib, Seaborn

## Project structure

```
flower-classification-transfer-learning/
├── notebooks/
│   └── transfer_learning_flowers102.ipynb   # full workflow with outputs
├── images/
│   ├── confusion_matrix_resnet50.png
│   └── sample_predictions.png
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## How to run

**Option 1: Google Colab (easiest).** Click the "Open In Colab" badge above, then choose Runtime > Change runtime type > GPU, and run all cells.

**Option 2: locally**

```bash
git clone https://github.com/varad-takale/flower-classification-transfer-learning.git
cd flower-classification-transfer-learning
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/transfer_learning_flowers102.ipynb
```

The dataset downloads automatically the first time you run the notebook. A GPU is strongly recommended: in the saved run, evaluating VGG16 on the test set took roughly 55 minutes.

## Limitations and next steps

- All three backbones were **frozen** (feature extraction only); there was no fine-tuning.
- No data augmentation was used.
- No from-scratch baseline was trained, so the exact benefit of transfer learning is not measured.
- Results come from a single run (no repeated runs or fixed seeds).

Planned improvements:
- [ ] Add data augmentation (random flip, rotation)
- [ ] Fine-tune the top layers of ResNet50 with a low learning rate and compare accuracy
- [ ] Add a from-scratch CNN baseline
- [ ] Try EfficientNet and plot training vs validation curves

## Author

**Varad Takale** | [GitHub](https://github.com/varad-takale)

## License

MIT, see [LICENSE](LICENSE).
