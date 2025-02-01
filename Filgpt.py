import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Prepare the Data ---

# Sample training sentences
sentences = [
    "hello how are you",
    "hello what is your name",
    "how is the weather today",
    "what time is it now"
]

# Tokenize sentences into words
all_words = " ".join(sentences).split()
vocab = sorted(set(all_words))
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

VOCAB_SIZE = len(vocab)
CONTEXT_SIZE = 2  # Use the last 2 words as context to predict the next word

# Build training sequences:
# For each sentence, for every consecutive sequence of CONTEXT_SIZE words,
# predict the next word.
train_inputs = []
train_targets = []
for sentence in sentences:
    tokens = sentence.split()
    if len(tokens) <= CONTEXT_SIZE:
        continue  # skip if sentence is too short
    for i in range(len(tokens) - CONTEXT_SIZE):
        context = tokens[i:i+CONTEXT_SIZE]
        target = tokens[i+CONTEXT_SIZE]
        # Convert words to indices
        train_inputs.append([word2idx[w] for w in context])
        train_targets.append(word2idx[target])

# Convert training data to tensors
X_train = torch.tensor(train_inputs, dtype=torch.long)  # shape: (num_samples, CONTEXT_SIZE)
Y_train = torch.tensor(train_targets, dtype=torch.long)   # shape: (num_samples)

# --- 2. Define the Model ---

class TinyLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_size):
        super(TinyLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # LSTM expects input shape (batch, seq_len, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # The fully connected layer outputs a prediction over the vocabulary
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.context_size = context_size

    def forward(self, x):
        # x shape: (batch, context_size)
        embeds = self.embed(x)  # shape: (batch, context_size, embed_dim)
        lstm_out, _ = self.lstm(embeds)  # shape: (batch, context_size, hidden_dim)
        # We take the output from the last time step
        last_output = lstm_out[:, -1, :]  # shape: (batch, hidden_dim)
        out = self.fc(last_output)  # shape: (batch, vocab_size)
        return out

# Hyperparameters
EMBED_DIM = 16
HIDDEN_DIM = 32

# Instantiate the model
model = TinyLM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, CONTEXT_SIZE)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- 3. Train the Model ---
EPOCHS = 1000
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)  # shape: (num_samples, vocab_size)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- 4. Interactive Prediction ---
def predict_next_word(model, input_text):
    """
    Takes an input string (e.g., a sentence), uses the last CONTEXT_SIZE words,
    and predicts the next word.
    """
    model.eval()
    tokens = input_text.lower().split()
    if len(tokens) < CONTEXT_SIZE:
        return f"Please provide at least {CONTEXT_SIZE} words for context."
    
    context = tokens[-CONTEXT_SIZE:]
    try:
        context_idx = torch.tensor([[word2idx[w] for w in context]], dtype=torch.long)
    except KeyError:
        return "One or more words not in vocabulary. Try using different words."
    
    with torch.no_grad():
        output = model(context_idx)  # shape: (1, vocab_size)
        predicted_idx = torch.argmax(output, dim=1).item()
        return idx2word[predicted_idx]

# --- 5. User Input Loop ---
print("\nEnter a sentence (at least 2 words) for prediction, or type 'exit' to quit.")
while True:
    user_input = input("\nYour input: ").strip()
    if user_input.lower() == "exit":
        break
    prediction = predict_next_word(model, user_input)
    print("Prediction:", prediction)
