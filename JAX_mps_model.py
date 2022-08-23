# coding: utf-8
import jax
import optax
import tqdm as tqdm
from DataLoader import get_data
from functions import mps_network_params, accuracy, compute_grad, loss

if __name__ == '__main__':
    size = 784
    chi = 15
    num_targets = 10
    batch_size = 64
    n_targets = 10
    num_epochs = 10
    learning_rate = 0.0001
    
    train_dataloader, trainacc_dataloader, test_dataloader = get_data()
    params = mps_network_params(size, chi, num_targets)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    for epoch in range(num_epochs):
        for x, y in tqdm.tqdm(train_dataloader):
            grads = compute_grad(params, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        train_x, train_y = next(iter(trainacc_dataloader))
        train_acc = accuracy(params, train_x, train_y)
            
        test_x, test_y = next(iter(test_dataloader))
        test_acc = accuracy(params, test_x, test_y)
        print("Epoch {}".format(epoch))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))
