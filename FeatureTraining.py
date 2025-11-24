from torch import nn, randn
import torch
from torch.utils.data import Dataset
import numpy as np

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
        self.hidden_layers = 1

        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.hidden_layers)
        params = dict(self.rnn.named_parameters())
        print("RNN parameters:", params.keys())

        # load the dataset from output files
        # self.data = torch.load(f"{data_dir}_data.pt")
        # self.data_tensors = torch.load(f"{data_dir}_audio_features.pt")
        # self.genre_labels_tensors = torch.load(f"{data_dir}_genre_labels.pt")
        # self.genre_labels = np.load(f"{data_dir}_genre_labels.npy", allow_pickle=True).tolist()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_data_item = self.data[idx]
        genre_label = self.genre_labels[idx]
        audio_data_tensor = self.data_tensors[idx]
        genre_label_tensor = self.genre_labels_tensors[idx]

        return genre_label_tensor, audio_data_tensor, genre_label, audio_data_item

    def forward(x, hx=None, batch_first=False):
        if batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()
        if hx is None:
            hx = torch.zeros(rnn.num_layers, batch_size, rnn.hidden_size)
        h_t_minus_1 = hx.clone()
        h_t = hx.clone()
        output = []
        for t in range(seq_len):
            for layer in range(rnn.num_layers):
                input_t = x[t] if layer == 0 else h_t[layer - 1]
                h_t[layer] = torch.tanh(
                    input_t @ params[f"weight_ih_l{layer}"].T
                    + h_t_minus_1[layer] @ params[f"weight_hh_l{layer}"].T
                    + params[f"bias_hh_l{layer}"]
                    + params[f"bias_ih_l{layer}"]
                )
            output.append(h_t[-1].clone())
            h_t_minus_1 = h_t.clone()
        output = torch.stack(output)
        if batch_first:
            output = output.transpose(0, 1)
        return output, h_t

def main():
    dataset = SongsFeatureTraining("songsdata-november-24")
    print(f"loaded {len(dataset)} items of data")
    print(f"example = {dataset[0]}")

if __name__ == "__main__":
    main()
