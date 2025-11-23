import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor

class SongsDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.data_tensors = []
        self.genre_labels = []
        self.genre_labels_tensors = []
        self.genres_uniq = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

        # load the dataset
        dataset = load_dataset(data_dir)
        train_data = dataset['train']
        print(len(train_data))

        # construct the AST feature extractor to pull features from the 
        feature_extractor = ASTFeatureExtractor(num_mel_bins=16)

        # build the audio and genre data by building the tensors from each audio item
        # and storing them with genre labels
        t = 0
        for data_item in train_data:
            t = t+1
            #print(t)
            audio_item = data_item['audio']
            genre_item = data_item['genre']

            audio_inputs = feature_extractor(audio_item["array"], sampling_rate=audio_item["sampling_rate"], return_tensors="pt")
            self.data.append(audio_item)
            self.data_tensors.append(audio_inputs.input_values)
            self.genre_labels.append(genre_item)
            if t==100: break

        for idx in range(len(self.genre_labels)):
            temp_tensor = torch.tensor([self.genre_labels[idx]], dtype=torch.long)
            self.genre_labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_data_item = self.data[idx]
        genre_label = self.genre_labels[idx]
        audio_data_tensor = self.data_tensors[idx]
        genre_label_tensor = self.genre_labels_tensors[idx]

        return genre_label_tensor, audio_data_tensor, genre_label, audio_data_item

# Load the dataset
dataset = SongsDataset("rpmon/fma-genre-classification")
print(f"loaded {len(dataset)} items of data")
print(f"example = {dataset[0]}")

'''
# Process with AST feature extractor
#print(len(inputs))

rnn = torch.nn.RNN(16, 200, 2)
h0 = torch.randn(2, 1024, 200)
output, hn = rnn(torch.input_values,h0)

print(output)
'''