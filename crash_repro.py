import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.18'

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import optax
import equinox as eqx
from flax.core import FrozenDict

import math
import numpy as np
import numpy.random as npr
from tqdm import tqdm
import sys

from immutables import Map

from functools import partial

from collections import deque

import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], device_type='GPU')

is_gpu = jax.default_backend() != 'cpu'

BATCH_SIZE = 64
SPLIT = True

def preprocess_mnist(image, label):
    image = tf.reshape(image, (-1,))
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - tf.constant(0.1307)) / tf.constant(0.3081)
    return image, label

def mnist_loader(dataset):
    dataset = dataset.map(preprocess_mnist)
    dataset_length = dataset.cardinality().numpy()
    dataset = dataset.shuffle(dataset_length)
    dataset = dataset.batch(BATCH_SIZE).prefetch(8).as_numpy_iterator()
    return dataset, dataset_length // BATCH_SIZE

def mnist_split_train_loader(split):
    if SPLIT:
        return mnist_loader(tf.data.Dataset.load(f'mnist_split/{split}'))
    return mnist_loader(tfds.load('mnist', split='train', as_supervised=True, shuffle_files=True))

def mnist_test_loader():
    return mnist_loader(tfds.load('mnist', split='test', as_supervised=True))

def linear(in_s, out_s, key = None):
    shape = (out_s, in_s)
    return random.normal(key, shape) * math.sqrt(2. / in_s)

def activation(x, params):
    match params['activation']:
        case 'ash':
            return ash(x, params)
        case 'hard_ash':
            return hard_ash(x, params)
        case 'topk_subtract':
            return topk_subtract(x, params)
        case 'topk_mask':
            return topk_mask(x, params)
        case 'lwta':
            return lwta(x, params)
        case 'elephant':
            return elephant(x, params)
        case 'relu':
            return jax.nn.relu(x)
        case 'swish':
            return jax.nn.swish(x)
        case 'sigmoid':
            return jax.nn.sigmoid(x)
        case 'hard_sigmoid':
            return jax.nn.hard_sigmoid(x)

def hard_ash(x, params):
    mean = jnp.mean(x)
    std = jnp.std(x)
    # return x * jax.nn.relu6(params['ash_alpha'] * (x - mean - params['ash_z_k'] * std) + 3.) / 6.0
    # is same as
    x = jnp.clip(x, a_min=0., a_max=2.) * jax.nn.hard_sigmoid(params['ash_alpha'] * (x - mean - params['ash_z_k'] * std))
    return x

def ash(x, params):
    mean = jnp.mean(x)
    std = jnp.std(x)
    return x * jax.nn.sigmoid(params['ash_alpha'] * (x - mean - params['ash_z_k'] * std))

def elephant(x, params):
    return 1. / (1 + jnp.abs(x / params['elephant_a']) ** params['elephant_d'])

def topk_subtract(x, params):
    k, _ = jax.lax.top_k(x, params['top_k'])
    v = k[-1]
    return jax.nn.relu(x - v)

def topk_mask(x, params):
    k, _ = jax.lax.top_k(x, params['top_k'])
    v = k[-1]
    return jnp.where(x >= v, x, jnp.zeros_like(x))

def lwta(x, params):
    x = jnp.reshape(x, (params['lwta_g'], -1))
    max = jnp.argmax(x, axis = -1)
    mask = jax.nn.one_hot(max, x.shape[-1])
    return jnp.reshape(mask * x, (-1))

def weight_norm(w, z, norm):
    if norm == False:
        return w @ z
    mean = jnp.mean(w, axis=1, keepdims=True)
    w = w - mean
    w = w / jnp.linalg.norm(w, axis=1, keepdims=True)
    return w @ z

class Model(eqx.Module):
    layers: list

    def __init__(self, key, params):
        sizes = [28*28] + ([params['layer_size']] * params['layer_count']) + [10]
        self.layers = []

        main_keys = random.split(key, len(sizes) - 1)
        for i, (in_c, out_c) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(linear(in_c, out_c, key = main_keys[i]))

    def __call__(self, x, params):
        z = x
        for i, l in enumerate(self.layers[:-1]):
            z = weight_norm(l, z, params['weight_norm'])
            z = activation(z, params)

        z = self.layers[-1] @ z
        return z

def train(model, params):
    match params['optimizer']:
        case 'rmsprop':
            optimizer = optax.rmsprop(params['learning_rate'], params['rmsprop_decay'])
        case 'adam':
            optimizer = optax.adam(params['learning_rate'], params['adam_b1'], params['adam_b2'])
        case 'adagrad':
            optimizer = optax.adagrad(params['learning_rate'])
        case 'sgd':
                optimizer = optax.sgd(params['learning_rate'])
        case 'sgdm':
            optimizer = optax.sgd(params['learning_rate'], momentum=params['sgdm_momentum'])
    
    @partial(jax.jit, static_argnums=1)
    # @eqx.filter_jit
    def step_classifier(model, params, opt_state, xs, ys):
        def classifier_loss(model, params, xs, ys):
            def go(x):
                return model(x, params)

            logits = jax.vmap(go)(xs)
            class_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=ys))

            return class_loss
        
        loss, g = jax.value_and_grad(classifier_loss)(model, params, xs, ys)

        updates, opt_state = optimizer.update(g, opt_state, model)
        updates = jtu.tree_map(lambda x: jnp.clip(x, a_min=-params['grad_clip'], a_max=params['grad_clip']), updates)
        model = optax.apply_updates(model, updates)

        return loss, model, opt_state
    
    def evaluation(model, params):
        num_classes = 10

        @partial(jax.jit, static_argnums=1)
        # @eqx.filter_jit
        def evaluate_step(model, params, xs, ys):
            def go(x):
                logits = model(x, params)
                return logits

            logits = jax.vmap(go)(xs)
            predicted_classes = jnp.argmax(logits, axis=1)

            correct_predictions_mask = predicted_classes == ys

            correct_counts = jnp.zeros(num_classes)
            for c in range(num_classes):
                correct_counts = correct_counts.at[c].set(jnp.sum(correct_predictions_mask & (predicted_classes == c)))

            total_counts = jnp.bincount(ys, length=num_classes)

            return correct_counts, total_counts

        test_loader, loader_length = mnist_test_loader()
        
        correct_counts = jnp.zeros(num_classes)
        total_counts = jnp.zeros(num_classes)

        for xs, ys in test_loader:
            correct, total = evaluate_step(model, params, xs, ys)
            correct_counts += correct
            total_counts += total
        
        accuracy = jnp.sum(correct_counts) / jnp.sum(total_counts)
        task_correct = jnp.sum(jnp.reshape(correct_counts, (5, 2)), axis=1)
        task_total = jnp.sum(jnp.reshape(total_counts, (5, 2)), axis=1)
        return accuracy.item(), task_correct / task_total

    losses = deque([0.], maxlen=100)

    opt_state = optimizer.init(model)

    for split in range(5 if SPLIT else 1):
        train_loader, loader_length = mnist_split_train_loader(split)
        for xs, ys in train_loader:

            loss, model, opt_state = step_classifier(model, params, opt_state, xs, ys)
            losses.append(loss)

            if np.isnan(loss):
                raise Exception(f'loss is nan')
        accuracy, per_task_accuracy = evaluation(model, params)
        if not is_sweep:
            log({
                'accuracy': accuracy,
                "train_loss": np.mean(losses),
                'task_0_accuracy': per_task_accuracy[0].item(),
                'task_1_accuracy': per_task_accuracy[1].item(),
                'task_2_accuracy': per_task_accuracy[2].item(),
                'task_3_accuracy': per_task_accuracy[3].item(),
                'task_4_accuracy': per_task_accuracy[4].item(),
            })

    return accuracy


# Wandb utils

def log(x):
    print(x)

def V(x):
    if is_sweep:
        return {
            'value': x
        }
    else:
        return x

def Vs(xs, manual_value=None):
    if is_sweep:
        return {
            'values': xs
        }
    else:
        if manual_value is not None:
            return manual_value
        return np.random.choice(xs).item()

def U(a, b, manual_value = None):
    if is_sweep:
        return {
            'distribution': 'uniform',
            'min': a,
            'max': b,
        }
    else:
        if manual_value is not None:
            return manual_value
        return a + (b - a) * np.random.random()

def logU(a, b, manual_value = None):
    if is_sweep:
        return {
            'distribution': 'log_uniform_values',
            'min': a,
            'max': b,
        }
    else:
        if manual_value is not None:
            return manual_value
        return a + (b - a) * np.random.random()


def lr_scale(x):
    return x * BATCH_SIZE / 256.

is_sweep = sys.argv[3] in ['t', 'true', 'sweep'] if len(sys.argv) >= 4 else False

# Configs

ash_config = {
    'ash_alpha': V(3.),
    # 'ash_alpha': Vs([3., 4.], manual_value=3.), # paper
    'ash_z_k': Vs([2.2, 2.3, 2.4], manual_value=2.3),
}

topk_config = {'top_k': Vs([32, 64, 96, 128, 256])}

activations = {
    'ash': ash_config,
    'hard_ash': ash_config,
    'elephant': {
        'elephant_a': Vs([0.02, 0.04, 0.08, 0.16], manual_value=0.16),
        'elephant_d': V(4.),
    },
    'topk_subtract': topk_config,
    'topk_mask': topk_config,
    'lwta': {
        'lwta_g': Vs([25,50,100])
    },
    'relu': {},
    'swish': {},
    'sigmoid': {},
    'hard_sigmoid': {},
}

optimizers = {
    'rmsprop' : {
        # 'rmsprop_decay': Vs([0.9991, 0.9992, 0.9993], manual_value=0.9993),
        # 'learning_rate': Vs([2e-6, 3e-6, 4e-6], manual_value=5.5e-6), #logU(4e-6, 6e-6),
        # paper
        'rmsprop_decay': Vs([0.999, 0.9991, 0.9992, 0.9993], manual_value=0.9993),
        'learning_rate': Vs([4e-6, 5e-6, 5.5e-6, 6e-6, 8e-6], manual_value=5.5e-6), #logU(4e-6, 6e-6),
    },
    'adam' : {
        'adam_b1': Vs([0.95, 0.98, 0.99], manual_value=0.98),
        'adam_b2': Vs([0.999, 0.9995], manual_value=0.9995),
        'learning_rate': Vs([8e-6, 1e-5, 1.5e-5], manual_value=1e-5), #logU(4e-6, 6e-6),
    },
    'adagrad' : {
        'learning_rate': Vs([3e-4, 5e-4, 1e-3], manual_value=1e-5), #logU(4e-6, 6e-6),
    },
    'sgd' : {
        'learning_rate': Vs([3e-4, 4e-4, 5e-4], manual_value=4e-4), #logU(4e-6, 6e-6),
    },
    'sgdm' : {
        'sgdm_momentum': Vs([0.99, 0.992, 0.994, 0.996], manual_value=0.998),
        'learning_rate': Vs([8e-6, 1e-5, 1.5e-5], manual_value=1e-5), #logU(4e-6, 6e-6),
    },
}

if not os.path.exists('./mnist_split'):
    for i in range(5):
        dataset = tfds.load('mnist', split='train', as_supervised=True, shuffle_files=True)
        dataset = dataset.filter(lambda x, y: (y // 2) == i).shuffle(1000)
        dataset.save(f'mnist_split/{i}')

chosen_activation_config = 'hard_ash'

for chosen_optimizer_config in optimizers.keys():

    sweep_config = {
        'method': 'grid',
        'name': f'{chosen_activation_config} : {chosen_optimizer_config}',
        'metric': {
        'name': 'accuracy_mean',
        'goal': 'maximize'
        },

        'parameters': {
            'weight_norm': V(1),
            'grad_clip': V(0.01),

            'layer_size': V(1000),
            'layer_count': V(1),
        }
    }
    sweep_config['parameters']['activation'] = V(chosen_activation_config)
    sweep_config['parameters'].update(activations[chosen_activation_config])

    sweep_config['parameters']['optimizer'] = V(chosen_optimizer_config)
    sweep_config['parameters'].update(optimizers[chosen_optimizer_config])

    key = random.PRNGKey(5678)

    NUM_RUNS = 1

    def run_train(params):
        accuracies = []
        for i in tqdm(range(NUM_RUNS)):
            seed = i + 12345
            key = random.PRNGKey(seed)
            tf.random.set_seed(seed)
            params = FrozenDict(params)
            model = Model(key, params)
            accuracies.append(train(model, params))
            std = np.std(accuracies)
            err = std / np.sqrt(len(accuracies))
            log({
                'accuracy_mean': np.mean(accuracies),
                'accuracy_error_bound': err * 1.96
            })
            if i > 0 and np.mean(accuracies) < 0.65:
                break

    params = sweep_config['parameters']
    print(params)

    model = Model(key, params)
    print(f'Model has {sum([np.size(x) for x in jtu.tree_leaves(model)]):,} params')

    run_train(params)

    jax.clear_caches()
        