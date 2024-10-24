# Ablation studies:
# 1. Change the number of epochs to 3 (increase by 1)
# 2. Change the batch size to 8 (increase by 4)

import os
from transformers import DistilBertTokenizer, TFAutoModelForSequenceClassification, TFDistilBertForSequenceClassification
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

# Initialize DistilBERT tokenizer to prepare the raw text into tokens that the model can understand
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Initialize the DistilBERT model for sequence classification
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Load the dataset by reading the text files from the 'train' and 'test' directories
# by iterating through the 'pos' and 'neg' subdirectories 
def load_data_from_directory(directory):
    # Initialize lists to store the texts and labels, texts being the movie reviews and labels being 0 for negative and 1 for positive
    texts = []
    labels = []
    # Iterate through the 'pos' and 'neg' subdirectories
    for label in ['pos', 'neg']:
        label_dir = os.path.join(directory, label)
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as file:
                    texts.append(file.read())
                    labels.append(1 if label == 'pos' else 0)
    return texts, labels

# Load train and test data 
train_texts, train_labels = load_data_from_directory('train')
test_texts, test_labels = load_data_from_directory('test')

# Set the maximum sequence length 
max_length = 128  
# Tokenize the texts with truncation and padding to the determined fixed max length
# Using the distilBERT tokenizer to tokenize the texts
train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=max_length)
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=max_length)

# Now that the texts are tokenized, convert the data into TensorFlow datasets
# which means converting the data into a format that TensorFlow can work with
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings),tf.convert_to_tensor(train_labels)))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings),tf.convert_to_tensor(test_labels)))

# Set the batch size, setting it low for runtime/memory sake 
batch_size = 8
# Shuffle the dataset to prevent patterns in the data from affecting the model
train_dataset = train_dataset.shuffle(len(train_texts)).batch(batch_size)
# Batch the test dataset
test_dataset = test_dataset.batch(batch_size)

# Compile the model
# Using the Adam optimizer with a learning rate of 3e-5
# Using SparseCategoricalCrossentropy as the loss function since the labels are integers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=3)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_dataset)

# Calculate precision and recall by predicting the test dataset
# and comparing the predictions to the true labels
y_pred_prob = model.predict(test_dataset)
y_pred = tf.argmax(y_pred_prob.logits, axis=1).numpy()
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
test_precision = precision_score(y_true, y_pred)
test_recall = recall_score(y_true, y_pred)

# Calculate F1 Score based on precision and recall
if(test_precision + test_recall == 0):
    f1_score = 0
else:
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)

# Print the results
print(f'Test loss: {test_loss}')
print('Test Accuracy:', test_acc)
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1 Score: {f1_score}')

# Plot the training and validation loss
train_loss = history.history['loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
