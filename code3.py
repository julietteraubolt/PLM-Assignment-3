import pandas as pd
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf

# Load pre-trained GPT-2 tokenizer  to encode the jokes into tokens
# and add the EOS token as the padding token so that the model knows where to stop
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Padding with EOS token
# Load the pre-trained GPT-2 model
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Load the dataset using pandas to handle the CSV format correctly
# since it is label, joke
jokes_file = 'data'  # Replace with your dataset path
jokes_df = pd.read_csv(jokes_file)

# # Print to check if the dataset is loaded correctly
# print(f"Dataset loaded: {jokes_df.head()}")

# Extract the 'Joke' column, cleant it like remove the surrounding quotation marks
jokes = jokes_df['Joke'].apply(lambda x: x.strip('"')).tolist()

# Print to ensure jokes are being extracted properly
print(f"First 5 jokes: {jokes[:5]}")

# Tokenize the jokes so that they can be fed into the model
def tokenize_joke(joke):
    # Tokenize the joke and return the input IDs and attention mask
    # which are needed for the model to understand the input
    tokens = tokenizer(joke, return_tensors='tf', max_length=128, padding='max_length', truncation=True)
    return tokens['input_ids'], tokens['attention_mask']

# Tokenize all the jokes and concatenate the input IDs and attention masks
# to create a single input tensor for the model
tokenized_jokes = [tokenize_joke(joke) for joke in jokes]
input_ids = tf.concat([tj[0] for tj in tokenized_jokes], axis=0)
attention_masks = tf.concat([tj[1] for tj in tokenized_jokes], axis=0)

# Prepare the labels for the model by shifting the input IDs by one position
def prepare_labels(input_ids):
    return tf.concat([input_ids[:, 1:], tf.fill((input_ids.shape[0], 1), tokenizer.eos_token_id)], axis=1)

# Get the target IDs for the model
target_ids = prepare_labels(input_ids)

# Convert to TensorFlow dataset so that it can be fed into the model
dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))
# Shuffle the dataset so that the model isn't affected by patterns
dataset = dataset.shuffle(1000)
# Batch the dataset so that the model trains on multiple jokes at once but 
# the batch size is small to fit in memory
dataset = dataset.batch(4)

# Compile the model with the Adam optimizer and sparse categorical crossentropy loss
# since the model is predicting the next token in the sequence for joke generation 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Train the model with for 4 epochs 
model.fit(dataset, epochs=4)

# Function to generate jokes based on the first three input words
def generate_joke(start_words, max_length=250, temperature=0.7, top_k=50, top_p=0.95):
    
    # Tokenize the input words, getting both input_ids and attention_mask
    inputs = tokenizer(start_words, return_tensors='tf', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']  # Get attention mask
    
    # Generate joke using better key characteristics like 
    # temperature, top_k, top_p, and do_sample which enables sampling
    # instead of greedy decoding and ensures the model is more creative
    # in generating jokes and not just repeating the same patterns 
    # while preventing the model from generating nonsensical jokes
    generated_outputs = model.generate(
        input_ids,                              # Input words
        max_length=max_length,                  # Maximum length of the joke
        num_return_sequences=1,                 # Generate  one joke
        pad_token_id=tokenizer.eos_token_id,    # Ensure padding uses EOS token
        temperature=temperature,                # Controls randomness in predictions
        top_k=top_k,                            # Limits to the top-k probable next words 
        top_p=top_p,                            # Nucleus sampling; picks from top-p cumulative probability
        repetition_penalty=1.2,                 # Discourages repeating tokens
        do_sample=True
    )

    # Decode and print the joke
    generated_joke = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
    print(f"Generated joke: {generated_joke}")
    return generated_joke

# 6 tests from data and random three words
start_words = "What did the"
print(generate_joke(start_words))

start_words = "My laptop is"
print(generate_joke(start_words))

start_words = "What's a pirates "
print(generate_joke(start_words))

start_words = "Blueberry boat crash"
print(generate_joke(start_words))

start_words = "Weather watch whale"
print(generate_joke(start_words))

start_words = "Surf link if"
print(generate_joke(start_words))