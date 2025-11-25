from torch import nn, randn
import torch
from torch.utils.data import Dataset
import numpy as np
import random

import Train

class SongsFeatureTraining(nn.Module):
    def __init__(self, data_dir):
        super(SongsFeatureTraining, self).__init__()

        self.data = []
        self.data_tensors = []
        self.genre_labels = []
        self.genre_labels_tensors = []
        self.genres_uniq = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
        self.input_size = 16
        self.hidden_size = 64
        self.hidden_layers = len(self.genres_uniq)

        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.hidden_layers)
        self.h2o = nn.Linear(self.hidden_size, len(self.genres_uniq))
        self.softmax = nn.LogSoftmax(dim=1)

        # load the dataset from output files
        # self.data = torch.load(f"{data_dir}_data.pt")
        self.data_tensors = torch.load(f"{data_dir}_audio_features.pt")
        self.genre_labels_tensors = torch.load(f"{data_dir}_genre_labels.pt")
        # self.genre_labels = np.load(f"{data_dir}_genre_labels.npy", allow_pickle=True).tolist()
    
    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)

        return output

    def label_from_output(output, output_labels):
        top_n, top_i = output.topk(1)
        label_i = top_i[0].item()
        return output_labels[label_i], label_i


def main():
    rnn = SongsFeatureTraining("songsdata-november-24")
    print("RNN Initialized: ", rnn)
    print("Example forward pass output: ", rnn.forward(torch.randn(1, 1, rnn.input_size)))
    print(Train.train(rnn, rnn.data_tensors))

if __name__ == "__main__":
    main()
