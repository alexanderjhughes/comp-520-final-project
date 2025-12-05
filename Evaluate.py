import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FeatureTraining import SongsFeatureRNN

model_dir = "rnn_model.pth"
validation_data_dir = "songsdata-november-24_validation_audio_features.pt"
validation_labels_dir = "songsdata-november-24_validation_genre_labels.pt"
genres_uniq = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SongsFeatureRNN()
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

def evaluate_model(model, device, validation_samples, labels):
    print("Evaluating Model\n\n")
    criterion = nn.NLLLoss()
    loss = 0
    correctly_predicted = 0
    total_samples = len(validation_samples)

    with torch.no_grad():
        for i in range(total_samples):
            current_sample = validation_samples[i].to(device)
            current_label = labels[i].to(device)

            result = model(current_sample)
            prediction = result.argmax(dim=1)
            #print('Prediction: ', prediction.item())
            #print('actual: ', current_label.item())
            if prediction.item() == current_label.item():
                correctly_predicted += 1


            loss += criterion(result, current_label).item()

    final_loss = loss / total_samples
    final_accuracy = correctly_predicted / total_samples

    return final_accuracy, final_loss

def main():
    model, device = load_model()
    validation_dataset, validation_labels = load_dataset()

    accuracy, loss = evaluate_model(model, device, validation_dataset, validation_labels)
    print(f"Validation Accuracy: {accuracy:.4f}\n")
    print(f"Validation Loss: {loss:.4f}\n\n")

if __name__ == "__main__":
    main()
