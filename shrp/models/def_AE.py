import torch.nn as nn
import torch
from einops import repeat
import math
from shrp.models.def_transformer import TransformerEncoder
import logging


class AE(nn.Module):
    def __init__(self, config):
        # TODO
        super(AE, self).__init__()
        # instanciate components
        i_dim = config.get("ae:i_dim", 201)
        d_model = config.get("ae:d_model", 512)
        nhead = config.get("ae:nhead", 8)
        num_layers = config.get("ae:num_layers", 6)
        lat_dim = config.get("ae:lat_dim", 16)
        windowsize = config.get("training::windowsize", 16)
        dropout = config.get("ae:dropout", 0.0)
        use_layer_embs = config.get("ae:use_layer_embs", False)
        use_layer_embs_enc_only = config.get("ae:use_layer_embs_enc_only", False)
        num_unique_layer_types = config.get("ae:num_unique_layer_types", 9)

        self.use_layer_embs = use_layer_embs
        self.use_layer_embs_enc_only = use_layer_embs_enc_only
        assert (
            d_model % nhead == 0
        ), f"invalid transformer config with d_model {d_model} and n_heads {nhead}"

        # mapping to token_dim
        self.tokenizer = nn.Linear(i_dim, d_model)
        # encoder
        if config.get("ae:transformer_type", "pytorch") == "pytorch":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
        elif config.get("ae:transformer_type", "pytorch") == "gpt2":
            self.transformer_encoder = TransformerEncoder(
                n_layer=num_layers,
                n_head=nhead,
                d_model=d_model,
                dropout=dropout,
                bias=False,
                causal=False,
                block_size=windowsize,
            )
        else:
            raise ValueError(
                f"invalid encoder type {config.get('ae:transformer_type')}"
            )
        # mapping from token_dim to lat_dim
        self.encoder_comp = nn.Linear(d_model, lat_dim)

        # decoder
        # mapping from token_dim to original dim
        use_hidden_layer = config.get("ae:decoder:use_hidden_layer", False)
        use_skip_connection = config.get("ae:decoder:use_skip_connection", False)
        if use_hidden_layer:
            if use_skip_connection:
                self.detokenizer = DetokenizerWithSkip(d_model, i_dim)
            else:
                self.detokenizer = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, i_dim),
                )
        else:
            self.detokenizer = nn.Linear(d_model, i_dim)

        # decoder is built of __ENcoder__ layers
        if config.get("ae:transformer_type", "pytorch") == "pytorch":
            decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.transformer_decoder = nn.TransformerEncoder(
                decoder_layer, num_layers=num_layers
            )
        elif config.get("ae:transformer_type", "pytorch") == "gpt2":
            self.transformer_decoder = TransformerEncoder(
                n_layer=num_layers,
                n_head=nhead,
                d_model=d_model,
                dropout=dropout,
                bias=False,
                causal=False,
                block_size=windowsize,
            )
        else:
            raise ValueError(
                f"invalid encoder type {config.get('ae:transformer_type')}"
            )
        # mapping from lat_dim to token_dim
        self.decoder_comp = nn.Linear(lat_dim, d_model)

        # position encoder
        if config.get("ae:pos_emb_type", None) == "functional":
            print("Using functional sinusoidal position embeddings")
            # position encoder using dynamic sinusoidal embeddings for 3D positional input
            self.pe = OptimizedSinusoidalPositionEmbeddings(
                embedding_dim=d_model, num_pos_dims=3
            )
        elif config.get("ae:pos_emb_type", None) == "quantized":
            print("Using quantized sinusoidal position embeddings")
            num_bins = config.get("ae:num_pos_buckets", 1024)
            # position encoder using dynamic sinusoidal embeddings for 3D positional input
            self.pe = QuantizedSinusoidalPositionEmbeddings(
                embedding_dim=d_model, num_pos_dims=3, num_bins=num_bins, cast_half=False
            )
        else:
            max_positions = config.get("ae:max_positions", [48, d_model])
            print(f"Using position embeddings with max_positions: {max_positions}")
            if config.get("ae:use_relative_pos", False) == True:
                print("Using learned relative position embeddings")
                self.pe = LearnedRelPosEmb(max_positions=max_positions, embedding_dim=d_model)
            else:
                print("Using learned position embeddings")
                self.pe = PositionEmbs(max_positions=max_positions, embedding_dim=d_model)

        # layer type embeddings
        if self.use_layer_embs:
            print(f"Using layer type embeddings with {num_unique_layer_types} types")
            self.layer_type_embs = LayerTypeEmbs(
                num_layer_types=num_unique_layer_types, embedding_dim=d_model
            )

        # projection head?
        # self.projection_head = ProjectionHead(
        #     d_model=lat_dim, nhead=4, num_layers=2, odim=30
        # )
        self.projection_head = SimpleProjectionHead(
            d_model=lat_dim, n_tokens=windowsize, odim=30
        )

        # dropout
        self.dropout = nn.Dropout(dropout)

        # taken from Kaparthy's GPT2 implementation:
        init_type = config.get("ae:weight_init_type", "normal")
        self.initialize_weights(init_type=init_type)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

        # aggregate embeddings
        self.aggregate_embeddings = config.get("ae:aggregate_embeddings", "mean")

    def initialize_weights(self, init_type="normal"):
        """
        Apply the weight initialization to all submodules using the given init_type.
        """
        def init_fn(module):
            self._init_weights(module, init_type=init_type)

        self.apply(init_fn)


    def _init_weights(self, module, init_type="normal"):
        """
        Initializes weights of the model based on module type and init_type
        Args:
            module: nn.Module
            init_type: str
        """
        if isinstance(module, torch.nn.Linear):
            if init_type == "normal":
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(module.weight)
            else:
                raise ValueError(f"invalid init type {init_type}")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, torch.nn.Embedding):
            # Embeddings are often initialized with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, torch.nn.LayerNorm):
            # LayerNorm layers typically use constant initialization
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    #
    def forward(self, x: torch.tensor, p: torch.tensor, mask=None, layer_type=None):
        """
        passes sequence of embeddings through encoder / decoder transformer
        Args:
            x: torch.tensor sequence of weight/channel tokens
            p: torch.tensor sequence of positions
            mask: optional torch.tensor mask for attention
        Returns:
            z: torch.tensor sequence of latent representations
            y: torch.tensor sequence of reconstructions
        """
        if self.use_layer_embs:
            z = self.forward_encoder(x, p, mask, layer_type)
        else:
            z = self.forward_encoder(x=x, p=p, mask=mask)
        zp = self.projection_head(z)
        if (self.use_layer_embs and not self.use_layer_embs_enc_only):
            y = self.forward_decoder(z, p, mask, layer_type)
        else:
            y = self.forward_decoder(z, p, mask=mask)
        return z, y, zp

    def forward_encoder(
        self, x: torch.tensor, p: torch.tensor, mask=None, layer_type = None
    ) -> torch.tensor:
        """
        Args:
            x: torch.tensor sequence of weight/channel tokens
            p: torch.tensor sequence of positions
            mask: optional torch.tensor mask for attention
            layer_type: optional torch.tensor sequence of layer types
        Returns:
            z: torch.tensor sequence of latent representations
        """
        # map weight tokens from input dim to d_model
        x = self.tokenizer(x)
        # add position embeddings
        x = self.pe(x, p)
        # add layer type embeddings
        if layer_type is not None:
            x = self.layer_type_embs(x, layer_type)

        # apply dropout
        x = self.dropout(x)
        # pass through encoder transformer
        x = self.transformer_encoder(x, mask=mask)
        # compress to latent dim
        x = self.encoder_comp(x)
        # return
        return x

    def forward_decoder(
        self, z: torch.tensor, p: torch.tensor, mask=None, layer_type = None
    ) -> torch.tensor:
        """
        Args:
            z: torch.tensor sequence of latent representations
            p: torch.tensor sequence of positions
            mask: optional torch.tensor mask for attention
            layer_type: optional torch.tensor sequence of layer types
        Returns:
            y: torch.tensor sequence of reconstructions
        """
        # map weight tokens from latent dim to d_model
        z = self.decoder_comp(z)
        # add position embeddings (again)
        z = self.pe(z, p)
        # add layer type embeddings
        if layer_type is not None:
            z = self.layer_type_embs(z, layer_type)
        # apply dropout
        z = self.dropout(z)
        # pass through decoder transformer
        z = self.transformer_decoder(z, mask=mask)
        # map back to original dim (so that it can be cast to checkpoint)
        z = self.detokenizer(z)

        return z

    def forward_embeddings(self, x: torch.tensor, p: torch.tensor) -> torch.tensor:
        """
        Args:
            x: torch.tensor sequence of weight/channel tokens
            p: torch.tensor sequence of positions
        Returns:
            z: torch.tensor sequence of latent representations
        """
        x = self.forward_encoder(x, p)
        # x = self.model.projection_head(x)
        # x = x.view(x.shape[0], -1)  # flatten
        if self.aggregate_embeddings == "mean":
            x = torch.mean(x, dim=1)  # average
        if self.aggregate_embeddings == "mean+std+quintiles":
            x1 = torch.mean(x, dim=1)  # average
            x2 = torch.std(x, dim=1)  # average
            # quintiles
            qts = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]).to(x.device)
            x3 = torch.quantile(x, qts, dim=1)
            # swap first and second axis
            x3 = x3.permute(1, 0, 2)
            x = torch.cat((x1.unsqueeze(dim=1), x2.unsqueeze(dim=1), x3), dim=1)
            x = x.view(x.shape[0], -1)  # flatten
        elif self.aggregate_embeddings == "flatten":
            x = x.view(x.shape[0], -1)  # flatten
        return x


class PositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.
    Attributes:
        posemb_init: positional embedding initializer.
        max_positions: maximum number of positions to embed.
        embedding_dim: dimension of the input embeddings.
    """

    def __init__(self, max_positions=[48, 256], embedding_dim=128):
        super().__init__()
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim
        if len(max_positions) == 2:
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)
            self.pe3 = None
        elif len(max_positions) == 3:
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)  # add 1 + 2
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)  # add 1 + 2
            self.pe3 = nn.Embedding(max_positions[2], embedding_dim // 2)  # cat 1+2 & 3

    def forward(self, inputs, pos):
        """Applies the AddPositionEmbs module.
        Args:
            inputs: Inputs to the layer, shape `(batch_size, seq_len, emb_dim)`.
            pos: Position of the first token in each sequence, shape `(batch_size,seq_len,2)`.
        Returns:
            Output tensor with shape `(batch_size, seq_len, emb_dim + 2)`.
        """
        assert (
            inputs.ndim == 3
        ), f"Number of dimensions should be 3, but it is {inputs.ndim}"
        assert pos.shape[2] == len(
            self.max_positions
        ), f"Position tensors should have as many demsions as max_positions. Pos shape: {pos.shape[2]}, MP: {self.max_positions}, LEN: {len(self.max_positions)}"
        assert (
            pos.shape[0] == inputs.shape[0]
        ), "Position tensors should have the same batch size as inputs"
        assert (
            pos.shape[1] == inputs.shape[1]
        ), "Position tensors should have the same seq length as inputs"

        pos_emb1 = self.pe1(pos[:, :, 0])
        pos_emb2 = self.pe2(pos[:, :, 1])
        if self.pe3 is not None:
            pos_emb3 = self.pe3(pos[:, :, 2])
            pos_emb = [pos_emb1 + pos_emb2, pos_emb3]
        else:
            pos_emb = [pos_emb1, pos_emb2]

        pos_emb = torch.cat(pos_emb, dim=2)

        out = inputs + pos_emb
        return out

class LayerTypeEmbs(nn.Module):
    def __init__(self, num_layer_types: int = 9, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_layer_types, embedding_dim)

    def forward(self, inputs, layer_type_ids: torch.Tensor) -> torch.Tensor:
        layer_emb = self.embedding(layer_type_ids)

        # reshape to match inputs
        if layer_emb.shape != inputs.shape:
            print(f"Reshaping: layer_emb {layer_emb.shape} → inputs {inputs.shape}")
            layer_emb = layer_emb.expand_as(inputs)  # Ensure same shape

        # Ensure dtype matches
        layer_emb = layer_emb.to(inputs.dtype)

        return inputs + layer_emb


class ProjectionHead(nn.Module):
    """
    Projection head: maps sequences of token embeddings and maps them to embeddings
    """

    def __init__(
        self, d_model: int = 512, nhead: int = 8, num_layers: int = 6, odim: int = 50
    ):
        super(ProjectionHead, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, odim, bias=False)
        self.comp_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, z: torch.tensor) -> torch.tensor:
        """
        Args:
            z: sequence of token embeddings [nbatch,token_window,token_dim]
        """
        # init compression token
        b, n, _ = z.shape
        copm_tokens = repeat(self.comp_token, "() n d -> b n d", b=b)
        z = torch.cat((copm_tokens, z), dim=1)
        # pass through
        z = self.encoder(z)
        # take only comp_token
        z = z[:, 0, :].squeeze()
        # pass through head
        z = self.head(z)
        # return
        return z


class SimpleProjectionHead(nn.Module):
    """
    Projection head: maps sequences of token embeddings and maps them to embeddings
    """

    def __init__(self, d_model: int = 512, n_tokens: int = 12, odim: int = 50):
        super(SimpleProjectionHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model * n_tokens, odim, bias=False),
            nn.LayerNorm(odim),
            nn.ReLU(),
            nn.Linear(odim, odim, bias=False),
            nn.LayerNorm(odim),
            nn.ReLU(),
        )

    def forward(self, z: torch.tensor) -> torch.tensor:
        """
        Args:
            z: sequence of token embeddings [nbatch,token_window,token_dim]
        """
        # avereage tokens
        # z = z.mean(dim=1)
        z = z.view(z.shape[0], -1)
        # pass through head
        z = self.head(z)
        # return
        return z

class DetokenizerWithSkip(nn.Module):
    def __init__(self, d_model, i_dim, hidden_dims=None):
        super(DetokenizerWithSkip, self).__init__()

        # Default hidden layers configuration if not provided
        if hidden_dims is None:
            hidden_dims = [d_model // 2, d_model // 4, d_model // 8]  # Add more layers as needed

        # Create the list of layers dynamically
        layers = []
        input_dim = d_model

        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.LeakyReLU())  # Use your preferred activation function
            input_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output projection
        self.proj_up = nn.Linear(input_dim, i_dim)

        # Skip projection
        self.skip_proj = nn.Linear(d_model, i_dim) if d_model != i_dim else nn.Identity()

    def forward(self, x):
        skip = self.skip_proj(x)  # Project the input if necessary
        x = self.hidden_layers(x)  # Pass through hidden layers
        x = self.proj_up(x)  # Project to output dimension
        x = x + skip  # Add the skip connection
        return x


class FunctionalSinusoidalPositionEmbeddings(nn.Module):
    """Applies sinusoidal positional embeddings with the assumption that the embedding dimension is even."""

    def __init__(self, embedding_dim: int, num_pos_dims: int = 3):
        super(FunctionalSinusoidalPositionEmbeddings, self).__init__()
        assert (
            embedding_dim % 2 == 0
        ), "Embedding dimension must be even to ensure compatibility with sin/cos structure."

        self.embedding_dim = embedding_dim
        self.num_pos_dims = num_pos_dims

    def _get_sinusoidal_embedding(self, positions, embedding_dim):
        """Generates sinusoidal embeddings dynamically for each positional dimension and sums them.

        Args:
            positions: Tensor of positions with shape `(batch_size, seq_len, num_pos_dims)`.
            embedding_dim: The size of the embedding to generate for each position.

        Returns:
            Tensor of shape `(batch_size, seq_len, embedding_dim)` with the summed sinusoidal embeddings.
        """
        batch_size, seq_len, num_pos_dims = (
            positions.size()
        )  # Extract batch size, seq length, and positional dimensions
        emb = torch.zeros(
            batch_size, seq_len, embedding_dim, device=positions.device
        )  # Initialize full embedding

        # Compute sinusoidal embeddings for each positional dimension and sum them up
        for i in range(num_pos_dims):
            position_indices = (
                positions[:, :, i].float().unsqueeze(-1)
            )  # Shape: (batch_size, seq_len, 1)
            div_term = torch.exp(
                torch.arange(0, embedding_dim, 2, device=positions.device).float()
                * (-math.log(10000.0) / embedding_dim)
            )

            # Create embedding for this position dimension
            pos_emb_dim = torch.zeros(
                batch_size, seq_len, embedding_dim, device=positions.device
            )

            # Apply sin to even indices and cos to odd indices
            pos_emb_dim[:, :, 0::2] = torch.sin(
                position_indices * div_term
            )  # Sine on even indices
            pos_emb_dim[:, :, 1::2] = torch.cos(
                position_indices * div_term
            )  # Cosine on odd indices

            # Sum the positional embeddings for each dimension
            emb += pos_emb_dim

        return emb

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Evaluates positional embeddings for inputs (handles both single sample and batch cases).

        Args:
            x: Inputs to the layer, shape `(batch_size, seq_len, emb_dim)` or `(seq_len, emb_dim)`.
            pos: Position tensor of shape `(batch_size, seq_len, num_pos_dims)` or `(seq_len, num_pos_dims)` for single samples.

        Returns:
            Output tensor with shape `(batch_size, seq_len, emb_dim)` or `(seq_len, emb_dim)` for single samples.
        """
        # If the input is not batched, add an extra batch dimension
        single_sample = x.ndim == 2  # Check if the input is a single sample
        if single_sample:
            x = x.unsqueeze(0)  # Add batch dimension
            pos = pos.unsqueeze(0)  # Add batch dimension for position

        # Compute positional embeddings and add them to input
        with torch.no_grad():
            pos_emb = self._get_sinusoidal_embedding(pos, self.embedding_dim)
        x_with_pos = x + pos_emb

        # If it was a single sample, squeeze the batch dimension back
        if single_sample:
            x_with_pos = x_with_pos.squeeze(0)

        return x_with_pos

class OptimizedSinusoidalPositionEmbeddings(nn.Module):
    """Memory-efficient sinusoidal positional embeddings with reduced intermediate allocations."""

    def __init__(self, embedding_dim: int, num_pos_dims: int = 3):
        super().__init__()
        assert embedding_dim % 2 == 0, "Embedding dimension must be even."

        self.embedding_dim = embedding_dim
        self.num_pos_dims = num_pos_dims

        # Precompute div_term once and register it as a buffer (so it's not a parameter but still moves with the model)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))
        self.register_buffer("div_term", div_term)

    def _get_sinusoidal_embedding(self, positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_pos_dims = positions.shape
        emb = torch.zeros(batch_size, seq_len, self.embedding_dim, device=positions.device, dtype=torch.float16)

        for i in range(num_pos_dims):
            position_indices = positions[:, :, i].unsqueeze(-1).float()
            sin_part = torch.sin(position_indices * self.div_term)
            cos_part = torch.cos(position_indices * self.div_term)
            emb[:, :, 0::2] += sin_part
            emb[:, :, 1::2] += cos_part

        return emb

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Applies sinusoidal embeddings efficiently to input."""
        single_sample = x.ndim == 2
        if single_sample:
            x = x.unsqueeze(0)
            pos = pos.unsqueeze(0)

        with torch.no_grad():  # Avoid tracking gradients for pos embeddings (saves memory in inference)
            pos_emb = self._get_sinusoidal_embedding(pos)

        x_with_pos = x + pos_emb

        return x_with_pos.squeeze(0) if single_sample else x_with_pos

class QuantizedSinusoidalPositionEmbeddings(nn.Module):
    """
    Memory-efficient sinusoidal positional embeddings that supports:
      - int positions (absolute indices) in arbitrary range
      - float positions (relative pos in [0,1]) with optional per-dimension bin-based quantization

    If `positions` are known to be in [0,1], we skip dynamic min/max and quantize
    directly within that fixed range.

    Arguments:
      embedding_dim (int): size of the positional embedding
      num_pos_dims (int): number of positional dimensions (default=3)
      num_bins (Optional[int]): if provided, floats in [0,1] will be quantized into this many bins
      cast_half (bool): if True, float positions are cast to half precision after bin-based quantization
    """

    def __init__(
        self,
        embedding_dim: int,
        num_pos_dims: int = 3,
        num_bins: int = 1000,
        cast_half: bool = False,
    ):
        super().__init__()
        assert embedding_dim % 2 == 0, "Embedding dimension must be even."

        self.embedding_dim = embedding_dim
        self.num_pos_dims = num_pos_dims
        self.num_bins = num_bins
        self.cast_half = cast_half

        # Precompute div_term and register as a buffer
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dim)
        )
        self.register_buffer("div_term", div_term)

    def _quantize_positions_per_dim(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Uniformly quantize positions per dimension *assuming [0,1] range* for each dimension.
        positions: [batch_size, seq_len, num_pos_dims] in float
        """
        # Clamp occasional floating drift beyond [0,1]
        positions = positions.clamp(min=0.0, max=1.0)

        # positions in [0,1] -> positions * (num_bins - 1)
        # round -> round( positions_normalized )
        # scale back -> x / (num_bins - 1)
        if self.num_bins is not None:
            scale_factor = float(self.num_bins - 1)
            positions = torch.round(positions * scale_factor) / scale_factor

        return positions

    def _get_sinusoidal_embedding(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: [batch_size, seq_len, num_pos_dims]
        Returns the sinusoidal embedding for each position.
        """
        batch_size, seq_len, num_dims = positions.shape

        # Case 1: positions is integer (absolute indices) -> cast to float
        if positions.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            positions = positions.float()

        # Case 2: positions is float -> optional bin-based quantization + optional half
        else:
            if self.num_bins is not None:
                positions = self._quantize_positions_per_dim(positions)
            if self.cast_half:
                positions = positions.half()

        # Prepare embedding tensor
        emb = torch.zeros(
            batch_size, seq_len, self.embedding_dim,
            device=positions.device, dtype=positions.dtype
        )

        # Expand div_term to match positions' dtype
        div_term = self.div_term.to(positions.dtype)

        # For each spatial dimension, add sine to even channels, cosine to odd channels
        for i in range(num_dims):
            # shape: [batch_size, seq_len, 1]
            position_indices = positions[:, :, i].unsqueeze(-1)
            sin_part = torch.sin(position_indices * div_term)
            cos_part = torch.cos(position_indices * div_term)
            emb[:, :, 0::2] += sin_part
            emb[:, :, 1::2] += cos_part

        return emb

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        x:   [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim] (single sample)
        pos: [batch_size, seq_len, num_pos_dims]  or [seq_len, num_pos_dims]

        Returns: x + positional_emb
        """
        single_sample = (x.ndim == 2)
        if single_sample:
            x = x.unsqueeze(0)
            pos = pos.unsqueeze(0)

        with torch.no_grad():
            pos_emb = self._get_sinusoidal_embedding(pos)

        # Cast pos_emb to match x's dtype before addition
        x_with_pos = x + pos_emb.to(x.dtype)

        return x_with_pos.squeeze(0) if single_sample else x_with_pos


class LearnedRelPosEmb(nn.Module):
    """Adds learned positional embeddings to the inputs.

    Attributes:
        max_positions (List[int]): maximum number of positions (bins) for each dimension.
        embedding_dim (int): dimension of the input embeddings.
    """

    def __init__(self, max_positions=[100000, 512, 100000], embedding_dim=128):
        """
        Args:
            max_positions: List of maximum positions for each dimension.
            embedding_dim: Dimension of the input (and output) embeddings.
        """
        super().__init__()
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim

        # If there are exactly 2 dimensions:
        if len(max_positions) == 2:
            # Each embedding is half the final dimension
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)
            self.pe3 = None

        # If there are exactly 3 dimensions:
        elif len(max_positions) == 3:
            # Each embedding is half the final dimension.
            # The original code sums pe1 & pe2 and concatenates pe3,
            # but keeps overall final size at embedding_dim.
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)
            self.pe3 = nn.Embedding(max_positions[2], embedding_dim // 2)

        else:
            raise ValueError(
                f"This example only handles 2 or 3 dims, got {len(max_positions)}."
            )

    def forward(self, inputs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Applies the PositionEmbs module.

        Args:
            inputs: Tensor of shape (batch_size, seq_len, embedding_dim).
            pos:    Tensor of shape (batch_size, seq_len, D), where D = len(max_positions).
                    Each coordinate is assumed in [0, 1], which will be scaled to
                    [0, max_positions[d] - 1] for dimension d.

        Returns:
            A tensor of shape (batch_size, seq_len, embedding_dim), which is
            `inputs + pos_embedding`.
        """
        # Basic checks
        assert inputs.ndim == 3, f"inputs must be 3D (batch, seq, emb). Got {inputs.shape}."
        assert pos.shape[0] == inputs.shape[0], "pos must have same batch size as inputs"
        assert pos.shape[1] == inputs.shape[1], "pos must have same seq length as inputs"
        assert pos.shape[2] == len(self.max_positions), (
            f"pos.shape[2] = {pos.shape[2]} does not match "
            f"len(max_positions) = {len(self.max_positions)}"
        )

        # Compute embeddings for each dimension
        pos_emb1 = self._get_scaled_embedding(self.pe1, pos[:, :, 0], self.max_positions[0])
        pos_emb2 = self._get_scaled_embedding(self.pe2, pos[:, :, 1], self.max_positions[1])

        if self.pe3 is not None:
            pos_emb3 = self._get_scaled_embedding(self.pe3, pos[:, :, 2], self.max_positions[2])
            # Original code: first two dims are summed, then the third dim is concatenated
            pos_emb = [pos_emb1 + pos_emb2, pos_emb3]
        else:
            # For 2D case: just put them in a list to cat below
            pos_emb = [pos_emb1, pos_emb2]

        # Concatenate the dimension embeddings along the embedding axis
        pos_emb = torch.cat(pos_emb, dim=2)  # shape: [batch, seq, embedding_dim]

        # Add positional embedding to inputs
        out = inputs + pos_emb
        return out

    @staticmethod
    def _get_scaled_embedding(
        embedding_layer: nn.Embedding,
        pos_coord: torch.Tensor,
        max_pos: int,
    ) -> torch.Tensor:
        """
        Scale and round pos_coord (in [0,1]) to an integer index in [0, max_pos-1],
        then look up from embedding_layer.

        Args:
            embedding_layer: nn.Embedding(max_pos, emb_size)
            pos_coord: Float tensor [batch_size, seq_len]
            max_pos: maximum position (integer) for this dimension

        Returns:
            Tensor of shape [batch_size, seq_len, emb_size]
        """
        # Clamp to [0,1] in case of small floating drift
        pos_coord = pos_coord.clamp_(0, 1)
        # Scale to [0, max_pos - 1], round to int
        scale = float(max_pos - 1)
        idx = (pos_coord * scale).round_().long()
        # Final safeguard clamp
        idx = idx.clamp_(0, max_pos - 1)
        # Lookup embedding
        return embedding_layer(idx)
