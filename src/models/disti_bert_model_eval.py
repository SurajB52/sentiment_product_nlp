#Warning: Do not Run this file without GPU (Google Colab: T4 12GB RAM, 15GB GPU)

#Libraries
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

#suppress some warnings
tf.get_logger().setLevel('ERROR')

# Load the trained model
model_path = '"H:\DS Projects\NLP Sentiment anlysis\data\models\sentiment_analysis_distilbert_model"'  # Adjust with your model path
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to tokenize validation data
def tokenize_data(tokenizer, sentences, max_length=128):
    return tokenizer(sentences, max_length=max_length, padding=True, truncation=True, return_attention_mask=True, return_tensors='tf')

# Function to load data in chunks for evaluation, similar to the training script
def load_data_in_chunks(file_path, chunk_size=10000):
    chunk_list = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Can select specific columns if needed
        chunk_list.append(chunk)
    return chunk_list

def data_generator(chunks, tokenizer, batch_size=32):
    for chunk in chunks:
        for i in range(0, len(chunk), batch_size):
            batch_data = chunk.iloc[i:i+batch_size]
            sentences = batch_data['reviewText'].tolist()
            labels = pd.Categorical(batch_data['sentiment']).codes  # Convert categorical data to numerical
            inputs = tokenize_data(tokenizer, sentences)
            yield ({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}, labels)
            
# Define your batch size and chunk size based on your system's capacity
BATCH_SIZE = 128
CHUNK_SIZE = 5000  # Adjust based on your system's memory

# Load test data
test_data_chunks = load_data_in_chunks('/content/drive/MyDrive/test_data_bert.csv', chunk_size=CHUNK_SIZE)  # Specify the correct path
print("Test data loaded...")

# Compile the model before evaluation
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)  # You can adjust the learning rate if needed
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # We use this loss as the model outputs logits
metrics = ['accuracy']  # Additional metrics can be added here

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print("Model is compiled...")

# Evaluation
evaluation_steps = sum(len(chunk) for chunk in test_data_chunks) // BATCH_SIZE
results = model.evaluate(
    data_generator(test_data_chunks, tokenizer, BATCH_SIZE),
    steps=evaluation_steps
)
print("Evaluation results (loss and accuracy):", results)

# Additional function to decode the predictions and true labels
def decode_predictions(pred_logits, true_labels):
    pred_labels = np.argmax(pred_logits, axis=1)  # Choose the label with the highest score
    return pred_labels, true_labels

# Predictions need to be gathered along with true labels
all_pred_logits = []
all_true_labels = []

# Iterate over the test data and make predictions
for inputs, labels in data_generator(test_data_chunks, tokenizer, BATCH_SIZE):
    predictions = model.predict(inputs)  # Get model predictions
    pred_labels, true_labels = decode_predictions(predictions.logits, labels)

    all_pred_logits.extend(predictions.logits)  # Store predictions
    all_true_labels.extend(true_labels)  # Store true labels

# Convert all true labels into a binary matrix, needed for the roc_auc_score calculation
from sklearn.preprocessing import label_binarize
binary_true_labels = label_binarize(all_true_labels, classes=[0, 1, 2])

# Calculate AUC-ROC score and ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):  # Assuming we have three sentiment classes 0, 1, 2
    fpr[i], tpr[i], _ = roc_curve(binary_true_labels[:, i], np.array(all_pred_logits)[:, i])
    roc_auc[i] = roc_auc_score(binary_true_labels[:, i], np.array(all_pred_logits)[:, i])

# Plotting the AUC-ROC curve
for i in range(3):
    plt.plot(fpr[i], tpr[i],
             label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Dashed baseline
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# Compute confusion matrix
conf_matrix = confusion_matrix(all_true_labels, np.argmax(all_pred_logits, axis=1))
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
print("\nClassification Report:")
print(classification_report(all_true_labels, np.argmax(all_pred_logits, axis=1), target_names=['Class 0', 'Class 1', 'Class 2']))



