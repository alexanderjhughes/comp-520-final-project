#!/usr/bin/env python3
import torch
import torch.nn as nn
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from FeatureTraining import SongsFeatureRNN

parser = argparse.ArgumentParser(
    description='Evaluate LSTM on Songs Dataset Features'
)

parser.add_argument(
    '-i',
    '--input_file_name',
    type=str,
    default="lstm_model.pth",
    help='Input file name for the trained model'
)

parser.add_argument(
    '-hl',
    '--hidden_layers_count',
    type=int,
    default=2,
    help='Number of hidden layers in the LSTM'
)

validation_data_dir = "songsdata-november-24_validation_audio_features.pt"
validation_labels_dir = "songsdata-november-24_validation_genre_labels.pt"
genres_uniq = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

def load_model(model_name, hidden_layers_count=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SongsFeatureRNN(hidden_layers_count)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.to(device)
    model.eval()
    print("Model has been loaded\n\n")
    return model, device

def load_dataset():
    validation_dataset = torch.load(validation_data_dir)
    validation_labels = torch.load(validation_labels_dir)
    return validation_dataset, validation_labels 
    print("Dataset loaded\n\n")

def evaluate_model(model, device, validation_samples, labels, input_file_name='lstm_model.pth'):
    print("Evaluating Model\n\n")
    criterion = nn.CrossEntropyLoss()
    correct_labels = []
    predictions = []
    loss = 0
    correctly_predicted = 0
    total_samples = len(validation_samples)

    with torch.no_grad():
        for i in range(total_samples):
            current_sample = validation_samples[i].to(device)
            current_label = labels[i].to(device)
            correct_labels.append(current_label.item())
            
            result = model(current_sample)
            prediction = result.argmax(dim=1)
            predictions.append(prediction.item())
            # print('Prediction: ', prediction.item())
            # print('actual: ', current_label.item())
            if prediction.item() == current_label.item():
                correctly_predicted += 1

            loss += criterion(result, current_label)

    matrix = confusion_matrix(correct_labels, predictions)
    #print(matrix)
    ConfusionMatrixDisplay(matrix, display_labels=genres_uniq).plot(xticks_rotation="vertical")
    formatted_input_file_name = input_file_name.replace('.pth', '')
    plt.savefig('confusion_matrices/confusion_matrix-' + formatted_input_file_name + '.png')
    plt.show()

    final_loss = loss / total_samples
    final_accuracy = correctly_predicted / total_samples

    return final_accuracy, final_loss

def main():
    args = parser.parse_args()
    print("Evaluating model from file:", args.input_file_name, 'with hidden layers:', args.hidden_layers_count, '\n\n')
    model, device = load_model(args.input_file_name, args.hidden_layers_count)
    validation_dataset, validation_labels = load_dataset()

    accuracy, loss = evaluate_model(model, device, validation_dataset, validation_labels, input_file_name=args.input_file_name)
    print(f"Validation Accuracy: {accuracy:.4f}\n")
    print(f"Validation Loss: {loss:.4f}\n\n")

if __name__ == "__main__":
    main()
