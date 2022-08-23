import numpy as onp
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad, lax
from jax import random
from torchvision.transforms import ToTensor, Resize, Lambda, Compose
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import torch

##can't use ones use identity matrices
def mps_network_params(size: int, chi: int, num_targets: int) -> list:
    """
    size: the amount of tensors in the mps and the number pixels in the images
    chi: the bond dimensions between the tensors
    num_targets: the number of categories an image can be put in
    
    Intializes the mps model that is going to classify images
    """
    
    mps = []
    first = jnp.array(onp.random.rand(2, chi) - 0.5)
    mps.append(first)
    
    middle = jnp.array(onp.random.rand(size-2, chi, 2, chi) - 0.5)
    mps.append(middle)
    #make middle a jax array
    '''
    for i in range(size-2):
        middle = middle.at[i].set(jnp.ones([chi,2,chi]))
    mps.append(middle)
    '''
    
    mps.append(jnp.zeros([chi,2,num_targets]))
    return mps

@jit
def scan(res: jnp.array, el: jnp.array) -> jnp.array:
    """
    res: what the function has contracted so far
    el: the new element that is going to be contracted into the tensor network 
    
    scan function during the contraction of all the tensors 
    """
    res = res @ el
    res /= jnp.linalg.norm(res)
    return res, res

@jit
def predict(params: list, img: np.array) -> jnp.array:
    """
    params: the mps network we are training
    img: the data
    
    predicts which category the image given falls into
    """
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
##rewrite so doesn't use dataloader
def accuracy(params: list, x: jnp.array, y: jnp.array) -> float:
    """
    params: the mps network we are training
    x: the data
    y: the target value
    
    To test the models accuracy
    """
    acc_total=0
    predicted = np.argmax(batched_predict(params, x), axis=1)
    acc_total += np.sum(predicted == y)
    return acc_total / x.shape[0]
    
@jit
def loss(params: list, images: np.array, target: int) -> float:
    """
    params: the mps network we are training
    images: the data
    target: the target value
    """
    preds = batched_predict(params, images)
    softmax_preds = jax.nn.softmax(preds)
    labels = jax.nn.one_hot(target, 10)
    return -jnp.sum(jnp.log2(preds) * target)


@jit
#optax adam optimizer instead
def compute_grad(params: list, x: np.array, y: int) -> float:
    """
    params: the mps network we are training
    x: the data
    y: the target value
    """
    return jax.grad(loss)(params, x, y)

#can have a seperate evaluate dataloader when testing for accuracy
