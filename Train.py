import torch, random, time
from torch import nn
import numpy as np
from tqdm import tqdm

def train(rnn, training_data, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    # Keep track of losses for plotting
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
        batches = np.array_split(batches, len(batches) // n_batch_size )

        print(f"Epoch {iter} out of {n_epoch}: {len(batches)} batches of size up to {n_batch_size}")
        progress_meter = tqdm(batches, desc=f"Epoch {iter}", unit="batch")

        batchNumberForPrinting = 0
        for batch in progress_meter:
            batch_loss = 0
            for i in batch: #for each example in this batch
                #print(training_data[i])
                (genre_label_tensor, feature_tensor) = training_data[i]
                output = rnn.forward(feature_tensor)
                #print("Output:", output)
                #print("Genre Label Tensor:", genre_label_tensor)
                loss = criterion(output, genre_label_tensor)
                #print("Loss:", loss.item())
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            progress_meter.set_postfix(loss=batch_loss.item() / len(batch))
            print(f"Batch Number: {batchNumberForPrinting}, Loss: {batch_loss.item() / len(batch)}")

            current_loss += batch_loss.item() / len(batch)
            batchNumberForPrinting += 1

        all_losses.append(current_loss / len(batches) )
        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0
    
    torch.save(rnn.state_dict(), "rnn_model.pth")

    return all_losses
