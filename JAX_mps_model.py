%load_ext autoreload
%autoreload 2
import numpy as onp
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from torchvision.transforms import ToTensor, Resize, Lambda, Compose
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import torch
import optax
from jax import lax
import tqdm as tqdm
from DataLoader import get_data

size = 784
chi = 15
num_targets = 10
batch_size = 64
n_targets = 10
num_epochs = 10
learning_rate = 0.0001

def mps_network_params(size: int, chi: int) -> list:
    mps = []
    mps.append(jnp.ones([2, chi]))
    
    middle = jnp.zeros([size-2, chi, 2, chi])
    #make middle a jax array
    for i in range(size-2):
        middle = middle.at[i].set(jnp.ones([chi,2,chi]))
    mps.append(middle)
    
    mps.append(jnp.ones([chi,2,num_targets]))
    return mps

@jit #make this faster

def scan(res, el):
    res = res @ el
    res /= jnp.linalg.norm(res)
    return res, res

@jit
def predict(params: list, img: np.array) -> jnp.array:
    contract = []
    
    #print(img[0].shape)
    contract.append(jnp.tensordot(img[0], params[0], [[0],[0]]))
        
    #einsum instead
    contract.append(jnp.einsum("ab,acbd->acd", img[1: -1], params[1]))
    
    contract.append(jnp.tensordot(img[len(img)-1], params[2], [[0],[1]]))
    result_init = contract[0]
    
    #jax.lax.scan instead of for
    final, result = lax.scan(scan, result_init, contract[1])
    out = final @ contract[2]
    
    return out

batched_predict = vmap(predict, in_axes=(None, 0))

@jit
def accuracy(params, dataloader):
    acc_total = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        images = np.array(data).reshape(data.size(0), 28*28)
        targets = one_hot(np.array(target), num_classes)

        target_class = np.argmax(targets, axis=1)
        predicted_class = np.argmax(batch_forward(params, images), axis=1)
        acc_total += np.sum(predicted_class == target_class)
    return acc_total/len(data_loader.dataset)

@jit
def loss_x(params, images, target):
    preds = batched_predict(params, images)
    preds = jnp.reshape(preds, [10,])
    logits = jax.nn.log_softmax(preds)
    #print(logits.shape)
    #raise ValueError
    labels = jax.nn.one_hot(target, num_targets)
    return -optax.softmax_cross_entropy(logits, labels)

@jit
def loss(params, images, target):
    preds = batched_predict(params, images)
    #print(preds.shape)
    #raise ValueError 
    #labels = jax.nn.one_hot(target, num_targets)
    
    return -jnp.mean(jnp.sum(preds * target, axis=1))

@jit
#optax adam optimizer instead
def compute_grad(params, x, y):
    return jax.grad(loss)(params, x, y)
  

train_dataloader, test_dataloader = get_data()
x, y = next(iter(train_dataloader))

params = mps_network_params(size, chi)
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

for epoch in range(num_epochs):
    for x, y in tqdm.tqdm(train_dataloader):
        z = jax.nn.one_hot(y, n_targets)
        grads = compute_grad(params, x, z)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
    train_acc = accuracy(params, training_dataloader)
    test_acc = accuracy(params, test_dataloader)
    print("Epoch {}".format(epoch))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
  
