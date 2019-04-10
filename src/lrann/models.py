# -*- coding: utf-8 -*-
"""
Different model architectures

Note: The BilinearNet class is more or less taken from Spotlight
"""
import torch
import torch.nn as nn


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class BaseModel(nn.Module):
    def __init__(self, n_users, n_items):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items


class BilinearNet(BaseModel):
    """
    Bilinear factorization representation.
    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    n_users (int): Number of users in the model.
    n_items (int): Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    biases (bool):
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.
    """

    def __init__(self, n_users, n_items, *, embedding_dim=32, biases=True,
                 user_embedding_layer=None, item_embedding_layer=None,
                 torch_seed=42, sparse=False):

        super().__init__(n_users, n_items)
        torch.manual_seed(torch_seed)

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(n_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(n_items, embedding_dim,
                                                   sparse=sparse)

        self.biases = biases
        if biases:
            self.user_biases = ZeroEmbedding(n_users, 1, sparse=sparse)
            self.item_biases = ZeroEmbedding(n_items, 1, sparse=sparse)
        else:
            self.user_biases = None
            self.item_biases = None

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        dot = user_embedding * item_embedding
        if dot.dim() > 1:  # handles case where embedding_dim=1
            dot = dot.sum(1)

        if self.biases:
            user_bias = self.user_biases(user_ids).squeeze()
            item_bias = self.item_biases(item_ids).squeeze()
            dot = dot + user_bias + item_bias

        return dot


class ResNet(BaseModel):
    def __init__(self, n_users, n_items, *, embedding_dim=32, biases=True, activation=None,
                 user_embedding_layer=None, item_embedding_layer=None,
                 torch_seed=42, sparse=False):

        super().__init__(n_users, n_items)
        torch.manual_seed(torch_seed)

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(n_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(n_items, embedding_dim,
                                                   sparse=sparse)

        self.biases = biases
        if biases:
            self.user_biases = ZeroEmbedding(n_users, 1, sparse=sparse)
            self.item_biases = ZeroEmbedding(n_items, 1, sparse=sparse)
        else:
            self.user_biases = None
            self.item_biases = None

        if activation is not None:
            self._activation = activation
        else:
            self._activation = torch.sigmoid

        self.h1 = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.h2 = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        dots = user_embedding * item_embedding

        res = self._activation(self.h1(dots))
        res = self.h2(res)
        res = self._activation(res)
        res = 1 + 0.5*(res - 0.5)
        dots = dots * res

        if dots.dim() > 1:  # handles case where embedding_dim=1
            dots = dots.sum(1)

        if self.biases:
            user_bias = self.user_biases(user_ids).squeeze()
            item_bias = self.item_biases(item_ids).squeeze()
            dots = dots + user_bias + item_bias

        return dots


class DeepNet(BaseModel):
    def __init__(self, n_users, n_items, *, embedding_dim=8, activation=None,
                 user_embedding_layer=None, item_embedding_layer=None,
                 torch_seed=42, sparse=False):

        super().__init__(n_users, n_items)
        torch.manual_seed(torch_seed)

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(n_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(n_items, embedding_dim,
                                                   sparse=sparse)
        if activation is not None:
            self._activation = activation
        else:
            self._activation = torch.sigmoid

        self._h1 = nn.Linear(2 * embedding_dim, embedding_dim * 4)
        self._h2 = nn.Linear(embedding_dim * 4, embedding_dim * 2)
        self._h3 = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        embedding = torch.cat([user_embedding, item_embedding], dim=1)

        # dispatch to allow calling the actual network manually
        return self._forward(embedding).squeeze()

    def _forward(self, input_):
        hidden = self._activation(self._h1(input_))
        hidden = self._activation(self._h2(hidden))
        out = self._h3(hidden)
        return out


class NaluNet(BaseModel):
    def __init__(self, n_users, n_items, *, embedding_dim=8,
                 user_embedding_layer=None, item_embedding_layer=None,
                 torch_seed=42, sparse=False):

        super().__init__(n_users, n_items)
        torch.manual_seed(torch_seed)

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(n_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(n_items, embedding_dim,
                                                   sparse=sparse)

        h1_dim = 2 * embedding_dim, embedding_dim * 4
        h2_dim = h1_dim[1], embedding_dim * 2
        h3_dim = h2_dim[1], 1
        self._h11 = nn.Linear(*h1_dim)
        self._h12 = nn.Linear(*h1_dim)
        self._h21 = nn.Linear(*h2_dim)
        self._h22 = nn.Linear(*h2_dim)
        self._h3 = nn.Linear(*h3_dim)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        embedding = torch.cat([user_embedding, item_embedding], dim=1)

        # dispatch to allow calling the actual network manually
        return self._forward(embedding).squeeze()

    def _forward(self, input_):
        hidden = torch.tanh(self._h11(input_)) * torch.sigmoid(self._h12(input_))
        hidden = torch.tanh(self._h21(hidden)) * torch.sigmoid(self._h22(hidden))
        out = self._h3(hidden)
        return out


class ResNetPlus(BaseModel):
    def __init__(self, n_users, n_items, *, embedding_dim=32, biases=True,
                 user_embedding_layer=None, item_embedding_layer=None,
                 torch_seed=42, sparse=False):

        super().__init__(n_users, n_items)
        torch.manual_seed(torch_seed)

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(n_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(n_items, embedding_dim,
                                                   sparse=sparse)

        self.biases = biases
        if biases:
            self.user_biases = ZeroEmbedding(n_users, 1, sparse=sparse)
            self.item_biases = ZeroEmbedding(n_items, 1, sparse=sparse)
        else:
            self.user_biases = None
            self.item_biases = None

        self.h2_shift = nn.Parameter(-5*torch.ones(2 * embedding_dim))
        self.h1_shift = nn.Parameter(-5*torch.ones(embedding_dim))
        self._h1 = nn.Linear(embedding_dim, 2 * embedding_dim)
        self._h2 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.w = nn.Parameter(-7*torch.ones(1))

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        dots = user_embedding * item_embedding

        res = torch.tanh(dots) * torch.sigmoid(self.h1_shift)
        res = torch.tanh(self._h1(res)) * torch.sigmoid(self.h2_shift)
        res = self._h2(res)
        dots = dots + torch.sigmoid(self.w) * res

        if dots.dim() > 1:  # handles case where embedding_dim=1
            dots = dots.sum(1)

        if self.biases:
            user_bias = self.user_biases(user_ids).squeeze()
            item_bias = self.item_biases(item_ids).squeeze()
            dots = dots + user_bias + item_bias

        return dots


class MoTBilinearNet(BaseModel):
    def __init__(self, n_users, n_items, *, embedding_dim=32, biases=True,
                 user_embedding_layer=None, item_embedding_layer=None,
                 torch_seed=42, sparse=False):

        super().__init__(n_users, n_items)
        torch.manual_seed(torch_seed)

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(n_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(n_items, embedding_dim,
                                                   sparse=sparse)

        self.biases = biases
        if biases:
            self.user_biases = ZeroEmbedding(n_users, 1, sparse=sparse)
            self.item_biases = ZeroEmbedding(n_items, 1, sparse=sparse)
        else:
            self.user_biases = None
            self.item_biases = None

        self.scaler_net = nn.Linear(embedding_dim, embedding_dim)
        self.scaler_w = nn.Parameter(-5*torch.ones(embedding_dim))

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        dots = user_embedding * item_embedding
        scaler_in = self.scaler_net(dots)
        scale = torch.sigmoid(self.scaler_w)*torch.tanh(scaler_in) + 1
        dots = scale * dots

        if dots.dim() > 1:  # handles case where embedding_dim=1
            dots = dots.sum(1)

        if self.biases:
            user_bias = self.user_biases(user_ids).squeeze()
            item_bias = self.item_biases(item_ids).squeeze()
            dots = dots + user_bias + item_bias

        return dots
