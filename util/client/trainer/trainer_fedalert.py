# util/client/trainer/trainer_fedalert.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import time
from .trainer_utils import read_energy
import gzip

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =====================================================================================
# LOCAL DATA LOADER
# =====================================================================================
def load_fashion_mnist_local(path):
    """Loads the Fashion MNIST dataset from local files."""
    files = [
        'train-labels-idx1-ubyte', 'train-images-idx3-ubyte',
        't10k-labels-idx1-ubyte', 't10k-images-idx3-ubyte'
    ]
    
    paths = [os.path.join(path, f) for f in files]

    with open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

# =====================================================================================
# MODEL DEFINITION
# =====================================================================================
def create_simple_cnn():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=5, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=5, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(10)
    ])
    return model

# =====================================================================================
# TrainerFedAlert Class
# =====================================================================================
class TrainerFedAlert:
    def __init__(self, id, name, args):
        self.id = id
        self.name = name
        self.args = args
        self.model = create_simple_cnn()
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.adaptive_lr = self.args.get("lr", 0.01)

        self.x_train, self.y_train, self.x_test, self.y_test = self._load_and_distribute_data()
        self.train_loader = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(buffer_size=len(self.y_train)).batch(self.args.get("batch_size", 32))
        
        self.loss_history = []
        self.local_threshold = float('inf')
        self.is_drifted = False
        self.stop_flag = False

    def _load_and_distribute_data(self):
        # --- MODIFIED: Use the correct local data path ---
        dataset_path = 'flw/data/FashionMNIST'
        (x_train_full, y_train_full), (x_test, y_test) = load_fashion_mnist_local(dataset_path)
        
        # Preprocess data
        x_train_full = np.expand_dims(x_train_full, -1).astype("float32") / 127.5 - 1.0
        x_test = np.expand_dims(x_test, -1).astype("float32") / 127.5 - 1.0

        distribution = self.args.get("data_distribution", "iid")
        num_clients = self.args.get("num_clients", 1)

        if distribution == "iid":
            indices = np.arange(len(x_train_full))
            np.random.seed(42)
            np.random.shuffle(indices)
            split_size = len(x_train_full) // num_clients
            client_indices = indices[self.id * split_size:(self.id + 1) * split_size]
            return x_train_full[client_indices], y_train_full[client_indices], x_test, y_test
        else: # non-iid
            num_shards, num_imgs = 200, 300
            idx_shard = list(range(num_shards))
            idxs = np.arange(num_shards * num_imgs)
            labels = y_train_full[:num_shards * num_imgs]
            
            idxs_labels = np.vstack((idxs, labels))
            idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
            idxs = idxs_labels[0, :]
            
            np.random.seed(self.id)
            shards_per_client = num_shards // num_clients
            rand_set = set(np.random.choice(idx_shard, shards_per_client, replace=False))
            
            client_indices = np.array([], dtype='int64')
            for rand in rand_set:
                client_indices = np.concatenate((client_indices, idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

            return x_train_full[client_indices], y_train_full[client_indices], x_test, y_test

    def apply_drift(self):
        if self.is_drifted: return
        drift_type = self.args.get("drift_type", "none")
        print(f"Client {self.id}: Applying '{drift_type}' drift.")
        
        dataset = self.train_loader.unbatch()
        new_images, new_labels = [], []

        if drift_type == "label_swap":
            for img, label in dataset:
                label_np = label.numpy()
                if label_np == 0: label_np = 1
                elif label_np == 1: label_np = 0
                new_images.append(img.numpy())
                new_labels.append(label_np)
        else:
            print(f"Client {self.id}: Drift type '{drift_type}' not implemented.")
            return

        new_dataset = tf.data.Dataset.from_tensor_slices((np.array(new_images), np.array(new_labels)))
        self.train_loader = new_dataset.shuffle(buffer_size=len(new_labels)).batch(self.args.get("batch_size", 32))
        self.is_drifted = True

    def train_model(self):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.adaptive_lr)
        loss_before = 0; num_batches = 0
        for data, target in self.train_loader:
            loss_before += self.criterion(target, self.model(data, training=False)).numpy()
            num_batches += 1
        loss_before /= num_batches if num_batches > 0 else 1

        start_time = time.time()
        for _ in range(self.args.get("epochs_per_client", 1)):
            for data, target in self.train_loader:
                with tf.GradientTape() as tape:
                    predictions = self.model(data, training=True)
                    loss = self.criterion(target, predictions)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.last_training_time = time.time() - start_time

        loss_after = 0; num_batches = 0
        for data, target in self.train_loader:
            loss_after += self.criterion(target, self.model(data, training=False)).numpy()
            num_batches += 1
        loss_after /= num_batches if num_batches > 0 else 1
        
        self.last_loss_improvement = loss_before - loss_after
        self.loss_history.append(self.last_loss_improvement)

    def eval_model(self):
        test_loader = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(self.args.get("batch_size", 32))
        self.model.compile(loss=self.criterion, metrics=['accuracy'])
        _, acc = self.model.evaluate(test_loader, verbose=0)
        return acc

    def all_metrics(self):
        is_alert = getattr(self, 'last_loss_improvement', 0) > self.local_threshold
        return {
            "accuracy": self.eval_model(),
            "loss_improvement": getattr(self, 'last_loss_improvement', 0),
            "is_alert": is_alert,
            "energy_consumption": read_energy(),
            "training_time": getattr(self, 'last_training_time', 0)
        }
    
    def set_local_threshold(self):
        if len(self.loss_history) > 1:
            mean, std = np.mean(self.loss_history), np.std(self.loss_history)
            self.local_threshold = mean + self.args.get("z_factor", 1.5) * std

    def get_weights(self): return self.model.get_weights()
    def update_weights(self, weights): self.model.set_weights(weights)
    def get_num_samples(self): return len(self.y_train)
    def get_stop_flag(self): return self.stop_flag
    def set_stop_true(self): self.stop_flag = True
    def set_args(self, args): self.args.update(args)