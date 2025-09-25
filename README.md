# Image Caption Generator with CNN & LSTM 🖼️📝

An advanced deep learning project that automatically generates descriptive captions for images using Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks.

## 🌟 Overview

This project combines computer vision and natural language processing to recognize the context of images and describe them in natural English. The model uses ResNet50 for feature extraction and LSTM networks for sequence generation, creating an end-to-end solution for automatic image captioning.

## 🎯 Key Features

- **CNN-based Feature Extraction**: Uses pre-trained ResNet50 for robust image feature extraction
- **LSTM Sequence Generation**: Employs LSTM networks for generating coherent text sequences
- **Encoder-Decoder Architecture**: Implements a sophisticated encoder-decoder model
- **BLEU Score Evaluation**: Validates model performance using industry-standard metrics
- **Batch Processing**: Efficient data processing to handle large datasets
- **Custom Image Captioning**: Generate captions for new, unseen images

## 📊 Dataset

The project uses the **Flickr8K dataset** containing:
- 8,091 images
- 40,455 captions (5 captions per image)
- Diverse image categories and scenarios

> **Note**: Smaller datasets like Flickr8K are used for faster training compared to larger datasets like Flickr30K or MSCOCO which can take weeks to train.

## 🏗️ Architecture

### CNN Feature Extractor
- **Model**: ResNet50 (pre-trained)
- **Output**: 2048-dimensional feature vectors
- **Preprocessing**: Image resizing to 224x224, normalization

### LSTM Caption Generator
- **Embedding Layer**: 256-dimensional word embeddings
- **LSTM Units**: 256 hidden units
- **Vocabulary Size**: 8,485 unique words
- **Max Sequence Length**: 35 words

### Model Architecture Flow
```
Image → ResNet50 → Feature Vector (2048) → Dense Layer (256)
                                                    ↓
Text Sequence → Embedding (256) → LSTM (256) → Add Layer
                                                    ↓
                                            Dense (256) → Softmax (8485)
```

## 📋 Requirements

```python
tensorflow>=2.12.0
keras>=2.12.0
numpy>=1.21.0
PIL>=8.3.0
matplotlib>=3.5.0
tqdm>=4.64.0
nltk>=3.7.0
pickle
os
re
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator
```

### 2. Install Dependencies
```bash
pip install tensorflow keras numpy pillow matplotlib tqdm nltk
```

### 3. Download Dataset
```bash
# Download Flickr8K dataset
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

# Extract files
unzip Flickr8k_Dataset.zip -d flickr8k/
unzip Flickr8k_text.zip -d flickr8k/
```

### 4. Project Structure
```
image-caption-generator/
├── flickr8k/
│   ├── Images/                 # Dataset images
│   └── captions.txt           # Image captions
├── model2/
│   ├── features.pkl           # Extracted image features
│   └── best_model.keras       # Trained model
├── test/                      # Test images folder
├── main.py                    # Main training script
└── README.md                  # Project documentation
```

## 💻 Usage

### 1. Feature Extraction
```python
# Load pre-trained ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
model = ResNet50(include_top=False, pooling='avg')

# Extract features from all images
features = {}
for img_name in os.listdir('flickr8k/Images'):
    # Load and preprocess image
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image.reshape((1, 224, 224, 3)))
    
    # Extract features
    feature = model.predict(image, verbose=0)
    features[img_name.split('.')[0]] = feature

# Save features
pickle.dump(features, open('model2/features.pkl', 'wb'))
```

### 2. Text Preprocessing
```python
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            # Convert to lowercase
            caption = caption.lower()
            # Remove special characters
            caption = re.sub(r'\s+', ' ', caption).strip()
            # Add start and end tokens
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
```

### 3. Model Training
```python
# Create model architecture
inputs1 = Input(shape=(2048,))  # Image features
inputs2 = Input(shape=(max_length,))  # Text sequences

# Image feature processing
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Text sequence processing
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# Decoder
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train model
for epoch in range(50):
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    model.fit(generator, steps_per_epoch=steps, epochs=1, verbose=1)
```

### 4. Generate Captions
```python
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += " " + word
    return in_text

# Generate caption for new image
caption = predict_caption(model, image_features, tokenizer, max_length)
print(f"Generated Caption: {caption}")
```

## 📈 Model Performance

### Training Results
- **Training Epochs**: 50
- **Batch Size**: 32
- **Training Images**: 7,282
- **Test Images**: 810

### Evaluation Metrics
- **BLEU-1 Score**: 0.528438 (52.84%)
- **BLEU-2 Score**: 0.306952 (30.70%)


## 🔧 Technical Details

### Data Processing Pipeline
1. **Image Preprocessing**: Resize to 224×224, normalize pixel values
2. **Text Preprocessing**: Lowercase, remove special characters, add start/end tokens
3. **Feature Extraction**: ResNet50 features (2048-dim vectors)
4. **Sequence Processing**: Tokenization, padding, categorical encoding

### Model Architecture Details
- **Total Parameters**: 16,405,361 (62.58 MB)
- **Trainable Parameters**: 5,468,453 (20.86 MB)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Activation**: ReLU (hidden), Softmax (output)

### Memory Optimization
- **Batch Processing**: Prevents memory overflow
- **Feature Caching**: Pre-computed CNN features saved as pickle
- **Generator Functions**: Efficient data loading during training

## 🎨 Example Usage

### For Dataset Images
```python
# Generate caption for dataset image
generate_caption("1001773457_577c3a7d70.jpg")
```

### For Custom Images
```python
# Generate caption for custom image
createcaption("your_image.jpg")
```

## 📊 Model Evaluation

### BLEU Score Calculation
```python
from nltk.translate.bleu_score import corpus_bleu

# Calculate BLEU scores
bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))

print(f"BLEU-1: {bleu1:.6f}")
print(f"BLEU-2: {bleu2:.6f}")
```

## 🚀 Future Improvements

- **Attention Mechanisms**: Implement visual attention for better focus
- **Beam Search**: Use beam search for better caption generation
- **Transformer Architecture**: Migrate to transformer-based models
- **Larger Datasets**: Train on MSCOCO or Flickr30K for better performance
- **Multi-modal Features**: Incorporate object detection features

## 📁 File Structure

```
├── feature_extraction.py      # CNN feature extraction
├── text_preprocessing.py      # Caption text cleaning
├── model_architecture.py     # LSTM model definition
├── data_generator.py         # Batch data generator
├── train_model.py           # Model training script
├── evaluate_model.py        # BLEU score evaluation
├── generate_captions.py     # Caption generation utilities
└── visualization.py         # Result visualization
```

## 🙏 Acknowledgments

- **Flickr8K Dataset** creators for providing the dataset
- **ResNet50** authors for the pre-trained CNN architecture

## 📚 References

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
