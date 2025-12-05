import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FeatureTraining import SongsFeatureTraining

model_dir = "rnn_model.pth"
validation_data_dir = "songsdata-november-24_validation_audio_features.pt"
validation_labels_dir = "songsdata-november-24_validation_genre_labels.pt"
genres_uniq = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SongsFeatureTraining()
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.to(device)
    model.eval()
    print("Model has been loaded\n\n")
    return model, device

def load_dataset():
    validation_dataset = torch.load(validation_data_dir)
    validation_labels = torch.load(validation_labels_dir)
    return validation_dataset, validation_labels 
    print("Dataset loaded\n\n")

def validation_accuracy(model, device, validation_samples, labels):
    print("Calculating accuracy")
    correctly_predicted = 0
    total_samples = len(validation_samples)

    with torch.no_grad():
        for i in range(total_samples):
            current_sample = validation_samples[i].to(device)
            current_label = labels[i].to(device)


            prediction = model(current_sample).argmax(dim=1)
            #print('Prediction: ', prediction.item())
            #print('actual: ', current_label.item())
            if prediction.item() == current_label.item():
                correctly_predicted += 1

    return correctly_predicted / total_samples

def validation_loss(model, device, validation_samples, labels):
    print("Calculating Loss\n\n")
    criterion = nn.NLLLoss()
    loss = 0

    with torch.no_grad():
        for i in range(len(validation_samples)):
            current_sample = validation_samples[i].to(device)
            current_label = labels[i].to(device)

            loss += criterion(model(current_sample), current_label).item()

    return loss / len(validation_samples)

def main():
    model, device = load_model()
    validation_dataset, validation_labels = load_dataset()

    accuracy = validation_accuracy(model, device, validation_dataset, validation_labels)
    print(f"Validation Accuracy: {accuracy:.4f}\n\n")

    loss = validation_loss(model, device, validation_dataset, validation_labels)
    print(f"Validation Loss: {loss:.4f}\n\n")

if __name__ == "__main__":
    main()
