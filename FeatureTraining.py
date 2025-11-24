from torch import nn, randn
import torch
from torch.utils.data import Dataset
import numpy as np

class SongsFeatureTraining(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.data_tensors = []
        self.genre_labels = []
        self.genre_labels_tensors = []
        self.genres_uniq = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

        # load the dataset from output files
        self.data = torch.load(f"{data_dir}_data.pt")
        self.data_tensors = torch.load(f"{data_dir}_audio_features.pt")
        self.genre_labels_tensors = torch.load(f"{data_dir}_genre_labels.pt")
        self.genre_labels = np.load(f"{data_dir}_genre_labels.npy", allow_pickle=True).tolist()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_data_item = self.data[idx]
        genre_label = self.genre_labels[idx]
        audio_data_tensor = self.data_tensors[idx]
        genre_label_tensor = self.genre_labels_tensors[idx]

        return genre_label_tensor, audio_data_tensor, genre_label, audio_data_item

def main():
    dataset = SongsFeatureTraining("songsdata-november-24")
    print(f"loaded {len(dataset)} items of data")
    print(f"example = {dataset[0]}")

if __name__ == "__main__":
    main()
