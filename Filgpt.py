import torch
import torch.nn as nn
import torch.optim as optim

# Sample training data (very simple)
sentences = [
    "hello how are you",
    "hello what is your name",
    "how is the weather today",
    "what time is it now"
]

# Tokenize and build vocabulary
words = set(" ".join(sentences).split())  # Unique words
word2idx = {word: i for i, word in enumerate(words)}
idx2word = {i: word for word, i in word2idx.items()}
VOCAB_SIZE = len(word2idx)
EMBED_DIM = 8  # Small embedding size
HIDDEN_DIM = 16  # Small hidden layer size

# Convert sentences to training pairs
train_data = []
for sentence in sentences:
    tokens = sentence.split()
    for i in range(len(tokens) - 1):
        input_word = word2idx[tokens[i]]
        target_word = word2idx[tokens[i + 1]]
        train_data.append((input_word, target_word))

# Convert to tensors
X_train = torch.tensor([x[0] for x in train_data], dtype=torch.long)
Y_train = torch.tensor([x[1] for x in train_data], dtype=torch.long)

# Define a small language model
class TinyLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TinyLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)  # Word embeddings
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)  # Simple RNN
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Output layer

    def forward(self, x):
        x = self.embed(x)  # Convert word index to embeddings
        x, _ = self.rnn(x.unsqueeze(0))  # Pass through RNN
        x = self.fc(x.squeeze(0))  # Convert hidden state to vocab size
        return x

# Initialize model, loss, and optimizer
model = TinyLM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Function to generate text from user input
def predict_next_word(model, word):
    model.eval()
    word_idx = word2idx.get(word, None)
    if word_idx is None:
        return "Word not in vocabulary. Try another."
    
    input_tensor = torch.tensor([word_idx], dtype=torch.long)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        return idx2word[predicted_idx]

# User input loop
while True:
    user_input = input("\nEnter a word: ").strip().lower()
    if user_input == "exit":
        break
    print(f"Prediction: {predict_next_word(model, user_input)}")
