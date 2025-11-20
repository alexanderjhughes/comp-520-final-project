from datasets import load_dataset
from torch import nn, randn

# Load the dataset
dataset = load_dataset("rpmon/fma-genre-classification")

# Access training data
train_data = dataset['train']

# Get an audio sample and its genre
audio = train_data[0]['audio']
genre = train_data[0]['genre']
print(audio)

# Process with AST feature extractor
from transformers import ASTFeatureExtractor
feature_extractor = ASTFeatureExtractor(num_mel_bins=16)
inputs = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
input_values = inputs.input_values
#print(len(inputs))

rnn = nn.RNN(16, 200, 2)
h0 = randn(2, 1024, 200)
output, hn = rnn(input_values,h0)

print(output)