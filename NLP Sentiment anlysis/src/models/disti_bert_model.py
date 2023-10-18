#Warning: Do not Run this file without GPU (Google Colab: T4 12GB RAM, 15GB GPU)

#Libraries
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TFDistilBertForSequenceClassification

# Set up GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data_in_chunks(file_path, chunk_size=10000):
    # Load data in smaller chunks
    chunk_list = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Filter out unnecessary columns here if needed
        chunk_list.append(chunk)
    return chunk_list  # You can also consider processing each chunk on-the-fly in the training loop

# Use a smaller chunk if you're facing memory issues
train_chunks = load_data_in_chunks('/content/drive/MyDrive/train_data_bert.csv', chunk_size=5000)
validation_chunks = load_data_in_chunks('/content/drive/MyDrive/validation_data_bert.csv', chunk_size=5000)
print("Data chunks are loaded...")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_data(tokenizer, sentences, max_length=128):
    return tokenizer(sentences, max_length=max_length, padding=True, truncation=True, return_attention_mask=True, return_tensors='tf')


def data_generator(chunks, tokenizer, batch_size=32):
    while True:
        for chunk in chunks:
            chunk = chunk.sample(frac=1)  # Shuffle the data
            for i in range(0, len(chunk), batch_size):
                batch_data = chunk.iloc[i:i+batch_size]
                sentences = batch_data['reviewText'].tolist()
                labels = pd.Categorical(batch_data['sentiment']).codes
                inputs = tokenize_data(tokenizer, sentences)
                yield ({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}, labels)


# Training parameters
BATCH_SIZE = 128  # Reduced batch size for memory constraints
EPOCHS = 2
STEPS_PER_EPOCH = sum(len(chunk) for chunk in train_chunks) // BATCH_SIZE
VALIDATION_STEPS = sum(len(chunk) for chunk in validation_chunks) // BATCH_SIZE
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
print("Model is compiled...")

# Train model
history = model.fit(
    data_generator(train_chunks, tokenizer, BATCH_SIZE),
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=data_generator(validation_chunks, tokenizer, BATCH_SIZE),
    validation_steps=VALIDATION_STEPS
)
print("Training complete...")

# Save model
model.save_pretrained("./sentiment_analysis_distilbert_model")
print('model saved...')
#Save model to path ".\NLP Sentiment anlysis\data\models"