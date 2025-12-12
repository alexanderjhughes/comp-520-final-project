import torch, random, time
from torch import nn
import numpy as np
from tqdm import tqdm

def train(rnn, training_data, n_epoch, learning_rate, output_file_name = 'gru_model.pth', n_batch_size = 64, report_every = 50, criterion = nn.CrossEntropyLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.to(device)
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        rnn.zero_grad() # clear the gradients

        # create some minibatches
        # we cannot use dataloaders because each of our names is a different length
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size )

        print(f"Epoch {iter} out of {n_epoch}: {len(batches)} batches of size up to {n_batch_size}")
        progress_meter = tqdm(batches, desc=f"Epoch {iter}", unit="batch")

        for batch in progress_meter:
            batch_loss = 0.0
            optimizer.zero_grad()
            for i in batch: #for each example in this batch
                #print(training_data[i])
                (genre_label_tensor, feature_tensor) = training_data[i]
                feature_tensor = feature_tensor.to(device)
                genre_label_tensor = genre_label_tensor.to(device)
                output = rnn(feature_tensor)
                #print("Output:", output)
                #print("Genre Label Tensor:", genre_label_tensor)
                loss = criterion(output, genre_label_tensor)
                #print("Loss:", loss.item())
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3.0)
            optimizer.step()
            

            progress_meter.set_postfix(loss=batch_loss.item() / len(batch))

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches) )
        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0
    
    torch.save(rnn.state_dict(), output_file_name)

    return all_losses
