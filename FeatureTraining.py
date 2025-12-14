#!/usr/bin/env python3
from torch import nn
import torch
import argparse
import Train

parser = argparse.ArgumentParser(
    description='Train LSTM on Songs Dataset Features'
)

parser.add_argument(
    '-e',
    '--epochs',
    type=int,
    default=40,
    help='Number of epochs to train the model'
)

parser.add_argument(
    '-l',
    '--learning_rate',
    type=float,
    default=0.005,
    help='Learning rate for training the model'
)

parser.add_argument(
    '-o',
    '--output_file_name',
    type=str,
    default="lstm_model.pth",
    help='Output file name for the trained model'
)

parser.add_argument(
    '-hl',
    '--hidden_layers_count',
    type=int,
    default=2,
    help='Number of hidden layers in the LSTM'
)

class SongsFeatureRNN(nn.Module):
    def __init__(self, hidden_layers_count = 2):
        super(SongsFeatureRNN, self).__init__()

        self.genres_uniq = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
        self.input_size = 128
        self.hidden_size = 64
        self.hidden_layers = hidden_layers_count
        self.dropout_rate = 0.2
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.attention = nn.Linear(self.hidden_size, 1)

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.hidden_layers, dropout=self.dropout_rate)
        self.h2o = nn.Linear(self.hidden_size, len(self.genres_uniq))
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor.permute(1, 0, 2), None)
        rnn_out = self.layernorm(rnn_out)
        attn_weights = torch.softmax(self.attention(rnn_out),dim=0)
        output = (attn_weights * rnn_out).sum(dim=0)
        output = self.h2o(output)
        output = self.softmax(output)

        return output

    def label_from_output(output, output_labels):
        top_n, top_i = output.topk(1)
        label_i = top_i[0].item()
        return output_labels[label_i], label_i

class SongsFeatureDataset():
    def __init__(self, data_dir):
        # load the dataset from output files
        # self.data = torch.load(f"{data_dir}_data.pt")
        data_tensors = torch.load(f"{data_dir}_audio_features.pt")
        genre_labels_tensors = torch.load(f"{data_dir}_genre_labels.pt")
        # self.genre_labels = np.load(f"{data_dir}_genre_labels.npy", allow_pickle=True).tolist()
        self.data = list(zip(genre_labels_tensors, data_tensors))

def main():
    args = parser.parse_args()
    dataset = SongsFeatureDataset("songsdata-november-24")
    rnn = SongsFeatureRNN(args.hidden_layers_count)
    print("LSTM Initialized: ", rnn)
    print('Starting Training...')
    print(Train.train(rnn, dataset.data, n_epoch=args.epochs, learning_rate=args.learning_rate, output_file_name=args.output_file_name))

if __name__ == "__main__":
    main()
