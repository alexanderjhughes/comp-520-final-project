#!/usr/bin/env python3
from torch import nn
import torch
import argparse
import Train

parser = argparse.ArgumentParser(
    description='Train GRU on Songs Dataset Features'
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
    default="gru_model.pth",
    help='Output file name for the trained model'
)

parser.add_argument(
    '-hl',
    '--hidden_layers_count',
    type=int,
    default=1,
    help='Number of hidden layers in the GRU'
)


# CNN-based model for feature classification
class SongsFeatureCNN(nn.Module):
    def __init__(self, hidden_layers_count=1):
        super(SongsFeatureCNN, self).__init__()
        self.genres_uniq = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
        self.input_size = 128
        self.num_classes = len(self.genres_uniq)
        self.hidden_layers_count = hidden_layers_count

        # Define the first conv layer
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_size
        out_channels = 128
        for i in range(hidden_layers_count):
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            self.conv_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
            out_channels = max(32, out_channels // 2)  # Decrease channels, but not below 32

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, self.num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        # line_tensor: (batch, seq_len, input_size) or (seq_len, input_size)
        if line_tensor.dim() == 2:
            line_tensor = line_tensor.unsqueeze(0)  # (1, seq_len, input_size)
        x = line_tensor.permute(0, 2, 1)  # (batch, input_size, seq_len)
        for i in range(0, len(self.conv_layers), 2):
            x = self.conv_layers[i](x)
            x = self.conv_layers[i+1](x)
            x = torch.relu(x)
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)        # (batch, channels)
        x = self.fc(x)
        x = self.softmax(x)
        return x

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
    rnn = SongsFeatureCNN(args.hidden_layers_count)
    print("GRU Initialized: ", rnn)
    print('Starting Training...')
    print(Train.train(rnn, dataset.data, n_epoch=args.epochs, learning_rate=args.learning_rate, output_file_name=args.output_file_name))

if __name__ == "__main__":
    main()
