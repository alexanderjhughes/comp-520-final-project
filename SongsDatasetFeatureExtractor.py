
from datasets import Audio, load_dataset
import ffmpeg
import torch
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
        self.validation_data = []
        self.validation_data_tensors = []
        self.validation_genre_labels = []
        self.validation_genre_labels_tensors = []

        # load the dataset
        dataset = load_dataset(data_dir)
        train_data = dataset['train']
        #disable default audio decoder
        train_data = train_data.cast_column("audio", Audio(decode=False))
        validation_data = dataset['validation']
        validation_data = validation_data.cast_column("audio", Audio(decode=False))

        # construct the AST feature extractor to pull features from the 
        feature_extractor = ASTFeatureExtractor(num_mel_bins=128)

        eachGenreCount = {}
        # build the audio and genre data by building the tensors from each audio item
        # and storing them with genre labels
        for data_item in train_data:
            audio_item = data_item["audio"]["bytes"]
            genre_item = data_item['genre']
            if genre_item not in eachGenreCount:
                eachGenreCount[genre_item] = 0
            eachGenreCount[genre_item] = eachGenreCount.get(genre_item, 0) + 1

            try:
                waveform = ffmpeg_decode(audio_item)
                audio_inputs = feature_extractor(waveform.numpy(), sampling_rate = 16000, return_tensors="pt")
                self.data.append(audio_inputs)
                self.data_tensors.append(audio_inputs.input_values)
                self.genre_labels.append(genre_item)
            except Exception as e: 
                print("SongsDataset Error:", e)

        for idx in range(len(self.genre_labels)):
            print(self.genre_labels[idx])
            print(self.genres_uniq)
            temp_tensor = torch.tensor([self.genre_labels[idx]], dtype=torch.long)
            self.genre_labels_tensors.append(temp_tensor)

        for val_data_item in validation_data:
            audio_item = val_data_item["audio"]["bytes"]
            genre_item = val_data_item['genre']

            try:
                waveform = ffmpeg_decode(audio_item)
                audio_inputs = feature_extractor(waveform.numpy(), sampling_rate = 16000, return_tensors="pt")
                self.validation_data.append(audio_inputs)
                self.validation_data_tensors.append(audio_inputs.input_values)
                self.validation_genre_labels.append(genre_item)
            except Exception as e: 
                print("SongsDataset Error:", e)

        for idx in range(len(self.validation_genre_labels)):
            print(self.validation_genre_labels[idx])
            print(self.genres_uniq)
            temp_tensor = torch.tensor([self.validation_genre_labels[idx]], dtype=torch.long)
            self.validation_genre_labels_tensors.append(temp_tensor)

        print("Genre distribution in training data:")
        for genre in eachGenreCount:
            print(f"Genre: {genre}, Count: {eachGenreCount[genre]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_data_item = self.data[idx]
        genre_label = self.genre_labels[idx]
        audio_data_tensor = self.data_tensors[idx]
        genre_label_tensor = self.genre_labels_tensors[idx]
        validation_audio_data_item = self.validation_data[idx]
        validation_genre_label = self.validation_genre_labels[idx]
        validation_audio_data_tensor = self.validation_data_tensors[idx]
        validation_genre_label_tensor = self.validation_genre_labels_tensors[idx]

        return genre_label_tensor, audio_data_tensor, genre_label, audio_data_item, validation_audio_data_item, validation_genre_label, validation_audio_data_tensor, validation_genre_label_tensor
    
    def save_all(self, prefix="songsdata"):
        self.save_tensors(prefix)
        np.save(f"{prefix}_genre_labels.npy", np.array(self.genre_labels, dtype=object))
        np.save(f"{prefix}_validation_genre_labels.npy", np.array(self.validation_genre_labels, dtype=object))

    def save_tensors(self, prefix="songsdata"):
        torch.save(self.data, f"{prefix}_data.pt")
        torch.save(self.data_tensors, f"{prefix}_audio_features.pt")
        torch.save(self.genre_labels_tensors, f"{prefix}_genre_labels.pt")
        torch.save(self.validation_data, f"{prefix}_validation_data.pt")
        torch.save(self.validation_data_tensors, f"{prefix}_validation_audio_features.pt")
        torch.save(self.validation_genre_labels_tensors, f"{prefix}_validation_genre_labels.pt")


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

if __name__ == '__main__':
    # Load the dataset
    dataset = SongsDataset("rpmon/fma-genre-classification")
    print(f"loaded {len(dataset)} items of data")
    print(f"example = {dataset[0]}")
    # Save all tensors
    dataset.save_all("songsdata-november-24")
