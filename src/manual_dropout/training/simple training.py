from itertools import product
from random import choices
from typing import List, Union, Tuple

import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import logging

from src.manual_dropout.model.feature import FeatureClassifier, FeatureRegressor

logging.basicConfig(level=logging.INFO)
from src.manual_dropout.model.vae import VAE
from src.utils.dataset import get_dataloader

np.set_printoptions(suppress=True)


def bce_loss(x, reconstruction):
    loss = torch.nn.BCELoss(reduction='sum')
    loss = loss(reconstruction, x)
    return loss


def feature_loss(feature_processors, parameters_pred, parameters_true):
    losses = [processor.calculate_loss(y_pred, y_true) for y_pred, y_true, processor in
              zip(parameters_pred, parameters_true.T, feature_processors)]
    loss = sum(losses)
    losses = [l.item() for l in losses]
    return loss, losses


def total_loss(images_true, images_pred,
               feature_processors,
               parameters_pred, parameters_true,
               betta: float = 1.):
    # loss 1
    bce = bce_loss(images_true, images_pred)
    # loss 2
    feature_l, losses = feature_loss(feature_processors=feature_processors,
                                     parameters_pred=parameters_pred,
                                     parameters_true=parameters_true)
    return bce + betta * feature_l, bce, betta * feature_l, losses


def train_model(autoencoder, optimizer, criterion, dataloader, epochs, device, feature_processors):
    logging.info(f'Start training')

    n_batches = len(dataloader)
    n_losses = 3

    train_losses = np.zeros((n_epochs, n_losses))
    train_separate_losses = np.zeros((n_epochs, len(feature_processors)))

    losses = []

    for epoch in tqdm(range(epochs)):
        autoencoder.train()
        train_losses_per_epoch = np.zeros(n_losses)
        train_separate_losses_per_epoch = np.zeros(len(feature_processors))

        for images_batch, parameters_batch in dataloader:
            # Batches to Cuda
            images_batch = images_batch.to(device)
            parameters_batch = parameters_batch.to(device)
            parameters_batch = parameters_batch[:, 1:]

            # Zero grad
            optimizer.zero_grad()

            # Forward pass
            images_pred, parameters_pred = autoencoder(images_batch)

            # Loss
            loss = criterion(images_true=images_batch, images_pred=images_pred,
                             feature_processors=feature_processors,
                             parameters_true=parameters_batch, parameters_pred=parameters_pred)
            loss[0].backward()

            for i in range(n_losses):
                train_losses_per_epoch[i] = loss[i].item()
            for i in range(len(feature_processors)):
                train_separate_losses_per_epoch[i] = loss[-1][i]

            optimizer.step()
        logging.info(f'Losses: \n{train_losses_per_epoch}, {train_separate_losses_per_epoch}')

        look_on_results(autoencoder, dataloader, device)
    return train_losses


def look_on_results(autoencoder, dataloader, device, n_to_show=6):
    autoencoder.eval()
    with torch.no_grad():
        for images_batch, parameters_batch in dataloader:
            images_pred, parameters_pred = autoencoder(images_batch.to(device))
            images_pred = images_pred.cpu().detach().numpy()
            ground_truth = images_batch.numpy()
            break

    plt.figure(figsize=(8, 20))
    for i, (gt, res) in enumerate(zip(ground_truth[:n_to_show], images_pred[:n_to_show])):
        plt.subplot(n_to_show, 2, 2 * i + 1)
        plt.imshow(gt.transpose(1, 2, 0), cmap='gray')
        plt.subplot(n_to_show, 2, 2 * i + 2)
        plt.imshow(res.transpose(1, 2, 0), cmap='gray')

    plt.show()


def save_model(autoencoder: VAE, path: str = './model.pt') -> None:
    torch.save(autoencoder, path)


def load_model(feature_processors, path: str = './model.pt', latent_dim: int = 1024) -> VAE:
    model: VAE = torch.load(path)
    return model


def plt_images(*images):
    plt.figure(figsize=(20, 8))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
    plt.show()


def plt_latent_operations(model, x1, x2):
    plt.figure(figsize=(40, 40))

    sampled_x1 = model.get_latent_vector(x1)
    sampled_x2 = model.get_latent_vector(x2)

    decoded_x1 = model.decode_latent(sampled_x1)
    decoded_x2 = model.decode_latent(sampled_x2)

    plt_images(x1, decoded_x1, x2, decoded_x2)
    k = 32

    variants = choices(list(product([True, False], repeat=5)), k=k)
    properties = variants
    for i in range(5):
        sample = [True] * 5
        sample[i] = False
        properties.append(sample)

    n_features = 5

    # for j in range((k % 5) // 1):
    #     properties = variants[j * 5: (j + 1) * 5]
    for i in range(n_features):
        result_vector = model.latent_operations(x1, x2, properties[i])
        plt.subplot(n_features, 3, 3 * i + 1)
        plt.imshow(x1.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
        plt.subplot(n_features, 3, 3 * i + 2)
        plt.imshow(result_vector.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
        plt.subplot(n_features, 3, 3 * i + 3)
        plt.imshow(x2.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
    plt.show()
    logging.info(f'{properties}')


def latent_plt(dataloader):
    image, values = next(dataloader)

    x1 = image[:1]
    x2 = image[1:2]

    x1 = x1.to(device)
    x2 = x2.to(device)

    sampled_x1 = autoencoder.get_latent_vector(x1)
    sampled_x2 = autoencoder.get_latent_vector(x2)

    plt_latent_operations(autoencoder, x1, x2)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    RESUME_TRAINING = False
    LOAD_MODEL = True

    bce = 1.
    kld = 1.
    feature = 1.
    latent_dim = 1024
    lr = 0.001
    n_epochs = 5
    image_size: Tuple[int, int, int] = (1, 64, 64)
    batch_size = 256

    shape_processor = FeatureClassifier(latent_dim, 3)
    x_processor = FeatureRegressor(latent_dim)
    y_processor = FeatureRegressor(latent_dim)
    size_processor = FeatureRegressor(latent_dim)
    rotate_processor = FeatureRegressor(latent_dim)
    feature_processors: List[Union[FeatureClassifier, FeatureRegressor]] = [processor.to(device) for processor in
                                                                            [shape_processor, x_processor,
                                                                             y_processor,
                                                                             size_processor, rotate_processor]]

    logging.info(f"Device: {device}")
    logging.info(f"Epochs: {n_epochs}")
    logging.info(f"Latent dim: {latent_dim}")
    logging.info(f'Creating model')

    # ------------------------------------------------------------
    # train
    # ------------------------------------------------------------
    #
    if LOAD_MODEL:
        autoencoder = load_model(feature_processors=feature_processors, latent_dim=latent_dim)
    else:
        autoencoder = VAE(latent_dim=latent_dim,
                          feature_processors=feature_processors)
    autoencoder = autoencoder.to(device)

    if RESUME_TRAINING:
        logging.info(f'Setting up dataloader')
        dataloader = get_dataloader('dsprites', batch_size=batch_size)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
        criterion = total_loss

        losses = train_model(autoencoder=autoencoder,
                             optimizer=optimizer,
                             criterion=criterion,
                             dataloader=dataloader,
                             epochs=n_epochs,
                             device=device,
                             feature_processors=feature_processors)
        save_model(autoencoder)

    # ------------------------------------------------------------
    # test
    # ------------------------------------------------------------

    dataloader = iter(get_dataloader('dsprites', batch_size=2))
    latent_plt(dataloader)
    latent_plt(dataloader)
    latent_plt(dataloader)
    latent_plt(dataloader)
    latent_plt(dataloader)
    latent_plt(dataloader)
    latent_plt(dataloader)

    print("Done")
