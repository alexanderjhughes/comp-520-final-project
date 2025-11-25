from torch import nn, randn
import torch
from torch.utils.data import Dataset
import numpy as np
import random

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

    def train(rnn, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss()):
        """
        Learn on a batch of training_data for a specified number of iterations and reporting thresholds
        """
        # Keep track of losses for plotting
        training_data = rnn.data_tensors
        current_loss = 0
        all_losses = []
        rnn.train()
        optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        start = time.time()
        print(f"training on data set with n = {len(training_data)}")

        for iter in range(1, n_epoch + 1):
            rnn.zero_grad() # clear the gradients

            # create some minibatches
            # we cannot use dataloaders because each of our names is a different length
            batches = list(range(len(training_data)))
            random.shuffle(batches)
            batches = np.array_split(batches, len(batches) //n_batch_size )

            for idx, batch in enumerate(batches):
                batch_loss = 0
                for i in batch: #for each example in this batch
                    (label_tensor, text_tensor, label, text) = training_data[i]
                    output = rnn.forward(text_tensor)
                    loss = criterion(output, label_tensor)
                    batch_loss += loss

                # optimize parameters
                batch_loss.backward()
                nn.utils.clip_grad_norm_(rnn.parameters(), 3)
                optimizer.step()
                optimizer.zero_grad()

                current_loss += batch_loss.item() / len(batch)

            all_losses.append(current_loss / len(batches) )
            if iter % report_every == 0:
                print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
            current_loss = 0

        return all_losses

def main():
    rnn = SongsFeatureTraining("songsdata-november-24")
    print("RNN Initialized: ", rnn)
    print("Example forward pass output: ", rnn.forward(torch.randn(1, 1, rnn.input_size)))
    print(rnn.train(rnn, rnn.data_tensors))

if __name__ == "__main__":
    main()
