import json
from joblib import load
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# =============================================================================
# SENTIMENT ANALYSIS PREDICTION SCRIPT - LEARNER VERSION
# =============================================================================
# This script demonstrates how to use a pre-trained LSTM model for sentiment analysis
# Key concepts covered: tokenization, sequence padding, neural network prediction

# Define the maximum sequence length (must match training configuration)
# TODO: Understand why we need a fixed sequence length for neural networks
MAX_PADDING_LENGTH = 33  # Fill in the appropriate length (hint: was 33 in original)

# =============================================================================
# STEP 1: LOAD PRE-TRAINED COMPONENTS
# =============================================================================

# Load class mappings (converts between class names and numeric labels)
# TODO: Understand the purpose of class mapping in classification tasks
with open("./models/class_mapping.json", "r", encoding="utf-8") as f:
    class_to_label_mapping = json.load(f)  # Load the JSON file

# Reverse the class_mapping to convert predictions back to class names
# TODO: Complete the dictionary comprehension to reverse key-value pairs
label_to_class_mapping = {v: k for k, v in class_to_label_mapping.items()}

def convert_prediction_to_label(prediction, mapping=label_to_class_mapping):
    """
    Convert numeric prediction to human-readable label
    
    Args:
        prediction: Numeric class prediction from model
        mapping: Dictionary mapping numbers to class names
    
    Returns:
        String label for the predicted class
    """
    # TODO: Use the mapping dictionary to get the class name
    # Hint: Use .get() method with a default value
    return mapping.get(prediction, "Unknown")

# Load the saved LSTM model
# TODO: Understand what LSTM stands for and why it's good for text analysis
lstm_model = load_model('./models/sentiment_lstm_model.keras')  # Fill in the correct model path

# Load the tokenizer (converts text to numbers)
# TODO: Research what tokenization means in NLP
tokenizer = load('./models/tokenizer.joblib')
  # Load the tokenizer from './models/tokenizer.joblib'

# =============================================================================
# STEP 2: PREDICTION FUNCTION
# =============================================================================

def predict_sentiment(text, model=lstm_model, tokenizer=tokenizer, max_padding_length=MAX_PADDING_LENGTH):
    """
    Predict sentiment of input text using pre-trained LSTM model
    
    Key steps:
    1. Convert text to numeric sequences (tokenization)
    2. Pad sequences to uniform length
    3. Get model prediction
    4. Convert prediction to readable label
    
    Args:
        text: Input text string to analyze
        model: Pre-trained LSTM model
        tokenizer: Fitted tokenizer for text-to-sequence conversion
        max_padding_length: Maximum sequence length for padding
    
    Returns:
        Predicted sentiment label as string
    """
    
    # STEP 2A: TOKENIZATION
    # Convert text to sequences of integers (each word becomes a number)
    # TODO: Understand why we need to convert text to numbers for neural networks
    sequences = tokenizer.texts_to_sequences([text])  # Fill in the correct method
    
    # STEP 2B: SEQUENCE PADDING
    # Ensure all sequences have the same length by padding/truncating
    # TODO: Research why neural networks require fixed-length inputs
    padded = pad_sequences(
        sequences, 
        maxlen=max_padding_length, 
        truncating='post'  # Truncate from the end if too long
    )
    
    # STEP 2C: MODEL PREDICTION
    # Get probability scores for each sentiment class
    # TODO: Understand what the model.predict() method returns
    prediction = model.predict(padded)  # Fill in the correct input
    
    # STEP 2D: PROCESS PREDICTION
    # Extract probabilities for the single input text
    probabilities = prediction[0]  # Get the first (and only) prediction
    
    # Find the class with highest probability
    # TODO: Research what np.argmax() does and why we use it
    max_prob_index = np.argmax(probabilities)
    
    # Convert numeric prediction to readable label
    predicted_label = convert_prediction_to_label(max_prob_index)
    
    return predicted_label

# =============================================================================
# STEP 3: TESTING THE MODEL
# =============================================================================

if __name__ == "__main__":
    # TODO: Add your own test examples and observe the predictions
    
    # Test Case 1: Positive sentiment example
    sample_text = "I absolutely loved this movie! It was fantastic."  # Add a positive text example
    print(f"Input Text: {sample_text}")
    prediction = predict_sentiment(sample_text)
    print(f"Predicted Sentiment: {prediction}")
    print()  # Empty line for readability
    
    # Test Case 2: Negative sentiment example  
    sample_text = "This was the worst experience I've ever had."  # Add a negative text example
    print(f"Input Text: {sample_text}")
    prediction = predict_sentiment(sample_text)  # Call the prediction function
    print(f"Predicted Sentiment: {prediction}")
    print()
    
    # TODO: Add more test cases to explore model behavior
    # Try examples with:
    # - Mixed emotions
    # - Very short texts
    # - Very long texts
    # - Neutral statements
    
    # BONUS TODO: Modify the function to also return confidence scores
    # BONUS TODO: Add error handling for empty or invalid inputs
