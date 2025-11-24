
from datasets import load_dataset
from torch import nn, randn
import ffmpeg
import torch
from datasets import load_dataset, Audio
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor
import numpy as np

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
        #disable default audio decoder
        train_data = train_data.cast_column("audio", Audio(decode=False))

        print(len(train_data))

        # construct the AST feature extractor to pull features from the 
        feature_extractor = ASTFeatureExtractor(num_mel_bins=16)

        # build the audio and genre data by building the tensors from each audio item
        # and storing them with genre labels
        t = 0
        for data_item in train_data:
            t = t+1
            #print(t)
            audio_item = data_item["audio"]["bytes"]
            genre_item = data_item['genre']

            try:
                waveform = ffmpeg_decode(audio_item)
                audio_inputs = feature_extractor(waveform.numpy(), sampling_rate = 16000, return_tensors="pt")
                self.data.append(audio_inputs)
                self.data_tensors.append(audio_inputs.input_values)
                self.genre_labels.append(genre_item)
            except Exception as e: 
                print("SongsDataset Error:", e)

            
            #if t==100: break

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


def ffmpeg_decode(audiobytes):
    try:
        #use pcm_s16le acodec and then normalize so that waveform is expected size for feature extractor, error can be ignored just want output
        output, error = (
            ffmpeg.input("pipe:", format="mp3").output("pipe:", format="wav", acodec="pcm_s16le", ar = 16000, ac = 1).run(input=audiobytes, capture_stdout=True, quiet = True)
        )
    except Exception as e:
        raise print("ffmpeg_decode Error:", e)

    #transform into proper format and normalize to use in AST feature extractor
    #convert pcmint16 byte data into number representation, convert to float32 for ASTFeatureExtractor, then divide by |minimum int16 value| to normalize s16 to [-1, 1]
    waveform_encoding = np.frombuffer(output, np.int16).astype(np.float32) / 32768.0

    #transform into pytorch tensor
    waveform_tensor = torch.tensor(waveform_encoding)
    
    return waveform_tensor

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