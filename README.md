# Sentiment Analysis on Tweets Using Deep Learning

## Objective
The goal of this project is to build and evaluate a deep learning model that classifies the sentiment of tweets as either positive or negative. This project utilizes natural language processing (NLP) and deep learning techniques to process and classify textual data.

## Dataset Overview
The dataset used in this project is the **Sentiment140 dataset**, consisting of 1.6 million labeled tweets. The dataset contains:
- **Labels**: 0 for negative sentiment and 4 for positive sentiment.
- **Time**: Timestamp of the tweet.
- **Date**: Human-readable date.
- **Username**: The Twitter handle of the user.
- **Tweet**: The content of the tweet.

### Data Distribution
- 800,000 negative tweets (`label = 0`)
- 800,000 positive tweets (`label = 4`)
- The dataset is balanced, providing a good foundation for model training.

## Data Preprocessing
Several preprocessing steps were applied to clean and prepare the dataset for model training:
1. **Tokenization**: The tweets are split into tokens (words).
2. **Sequence Padding**: All sequences are padded to ensure a uniform input size.
3. **Stopword Removal**: Common words (e.g., "the," "is") are removed.
4. **Lowercasing**: Tweets are converted to lowercase to eliminate case sensitivity.
5. **Data Splitting**: The data is split into 80% training and 20% testing using `train_test_split`.
6. **Text Tokenization**: Keras’s Tokenizer is used to convert words into integer indices.

## Model Architecture
The core of the model is an LSTM-based neural network with the following structure:
1. **Embedding Layer**: Converts input sequences to dense vectors.
2. **LSTM Layer**: Captures long-term dependencies in the sequences, with 128 units.
3. **Dense Layer with Dropout**: A fully connected layer with 128 neurons and 50% dropout.
4. **Output Layer**: A Dense layer with a sigmoid activation function for binary classification.

### Model Summary:
- **Embedding Layer**: Transforms tweets into dense word vectors.
- **LSTM Layer**: Learns the sequential relationships between words.
- **Dense Layer**: Adds transformations to the learned features.
- **Sigmoid Output Layer**: Produces a probability score for sentiment classification.

## Training and Evaluation
- **Optimizer**: The Adam optimizer is used.
- **Loss Function**: Binary cross-entropy for binary classification.
- **Training**: The model was trained for 10 epochs with early stopping to prevent overfitting.
- **Evaluation Metrics**: Accuracy, Precision, Recall, and Confusion Matrix.

### Results
- **Accuracy**: The model achieves approximately **85% accuracy** on the test set.
- **Confusion Matrix**: Shows a balanced performance in predicting both positive and negative sentiments.

## Challenges and Improvements
### Challenges
- **Noisy Data**: Tweets often contain slang, abbreviations, and emoticons that are difficult to process.
- **Out-of-Vocabulary Words**: Words not seen during training may be difficult for the model to classify.

### Future Improvements
- **Data Augmentation**: Using synonym replacement or back-translation could improve model robustness.
- **Transformer Models**: Models like BERT or GPT could better capture context and sentiment nuances.
- **Sentiment Intensity**: Moving from binary classification to predicting the intensity of sentiment could provide more granular insights.

## Conclusion
This project demonstrates the application of LSTM-based models to sentiment analysis of tweets. The model achieves strong performance but could be further improved by addressing noisy data and incorporating more advanced models like Transformers.


