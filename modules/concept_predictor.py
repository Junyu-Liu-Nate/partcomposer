import torch
import torch.nn as nn
import torch.nn.functional as F
import math

############################## Concept predictors ##############################
class ConceptClassifier(nn.Module):
    """
    A simple multi-label classification network to predict which concept tokens
    appear in a given latent. Typically, use BCEWithLogitsLoss.
    """
    def __init__(self, latent_channels=4, latent_size=64, out_dim=8, hidden_dim=256):
        super().__init__()
        """
        Args:
            latent_channels: Number of channels in the latents (often 4 for SD).
            latent_size: Height/width of the latent (often 64 for SD).
            out_dim: Number of concepts (e.g., total tokens like <asset0>.. <asset7>).
            hidden_dim: Hidden dimension for the fully connected layer(s).
        """
        self.conv1 = nn.Conv2d(latent_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # After these, shape ~ (B, 64, latent_size/8, latent_size/8), i.e. (B, 64, 8, 8) if latent_size=64

        self.fc = nn.Linear(64 * (latent_size // 8) * (latent_size // 8), hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        Args:
            x: Latents of shape (B, latent_channels, latent_size, latent_size)
        Returns:
            logits of shape (B, out_dim) for multi-label classification
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))
        logits = self.out(x)
        return logits

class ConceptClassifierSegmenter(nn.Module):
    """
    A simple multi-label classification + location network to predict:
      (1) which concept tokens appear in a given latent (multi-label classification), and
      (2) where each concept appears via a per-concept mask (logits_mask).
    """
    def __init__(self, latent_channels=4, latent_size=64, out_dim=8, hidden_dim=256):
        super().__init__()
        """
        Args:
            latent_channels: Number of channels in the latents (often 4 for SD).
            latent_size: Height/width of the latent (often 64 for SD).
            out_dim: Number of concepts (e.g. <asset0>..<asset7>).
            hidden_dim: Hidden dimension for the fully-connected layers.
        """

        self.conv1 = nn.Conv2d(latent_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # After these, shape ~ (B, 64, latent_size/8, latent_size/8)
        # i.e. (B, 64, 8, 8) if latent_size=64

        self.fc = nn.Linear(64 * (latent_size // 8) * (latent_size // 8), hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

        #################################################################
        #  A "location head" to produce per-concept mask logits
        #         shape => (B, out_dim, 8, 8) which we then upsample back to (64, 64)
        #################################################################
        self.mask_conv = nn.Conv2d(64, out_dim, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        #################################################################

    def forward(self, x):
        """
        Args:
            x: Latents of shape (B, latent_channels, latent_size, latent_size)
        Returns:
            logits_cls:  (B, out_dim) for multi-label classification
            logits_mask: (B, out_dim, latent_size, latent_size) per-pixel location logits
                         (e.g. 64x64 if latent_size=64)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))        # shape (B, 64, 8, 8) if latent_size=64

        #################################################################
        # location head
        #################################################################
        mask_logits_8x8 = self.mask_conv(x)  # (B, out_dim, 8, 8)
        logits_mask = self.upsample(mask_logits_8x8)  # (B, out_dim, 64, 64)
        #################################################################

        # Classification head
        x_flat = x.view(x.size(0), -1)   # flatten
        x_fc = F.relu(self.fc(x_flat))
        logits_cls = self.out(x_fc)

        #################################################################
        # return both classification logits + mask logits
        #################################################################
        return logits_cls, logits_mask

############################## Time-conditioned Concept Predictors ##############################
class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
            
        # timesteps: (B,) tensor or scalar; we assume float values in [0, num_train_timesteps]
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # shape: (B, embedding_dim)

class ConceptClassifierWithTime(nn.Module):
    """
    A multi-label classification network to predict which concept tokens appear in a latent,
    conditioned on the denoising timestep.
    """
    def __init__(self, latent_channels=4, latent_size=64, out_dim=8, hidden_dim=256, time_embedding_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # After conv layers, shape is (B, 64, latent_size/8, latent_size/8)
        conv_output_dim = 64 * (latent_size // 8) * (latent_size // 8)
        self.time_emb = TimeEmbedding(time_embedding_dim)
        # Concatenate flattened conv features with time embedding.
        self.fc = nn.Linear(conv_output_dim + time_embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, timestep):
        """
        Args:
            x: latent tensor of shape (B, latent_channels, latent_size, latent_size)
            timestep: a tensor of shape (B,) or a scalar representing the current diffusion timestep.
        Returns:
            logits: Tensor of shape (B, out_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x_flat = x.view(x.size(0), -1)
        t_emb = self.time_emb(timestep)  # shape: (B, time_embedding_dim)
        x_cat = torch.cat([x_flat, t_emb], dim=1)
        x_fc = F.relu(self.fc(x_cat))
        logits = self.out(x_fc)
        return logits

class ConceptClassifierSegmenterWithTime(nn.Module):
    """
    A multi-label classification + location network to predict:
      (1) which concept tokens appear in a given latent (multi-label classification), and
      (2) where each concept appears via a per-concept mask (logits_mask),
    conditioned on the current diffusion timestep.
    """
    def __init__(self, latent_channels=4, latent_size=64, out_dim=8, hidden_dim=256, time_embedding_dim=32):
        super().__init__()
        """
        Args:
            latent_channels: Number of channels in the latents (often 4 for SD).
            latent_size: Height/width of the latent (often 64 for SD).
            out_dim: Number of concepts (e.g. <asset0>..<asset7>).
            hidden_dim: Hidden dimension for the fully-connected layers.
            time_embedding_dim: Dimensionality of the time embedding.
        """
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(latent_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # After conv layers, shape is (B, 64, latent_size/8, latent_size/8)

        # Time embedding module
        self.time_emb = TimeEmbedding(time_embedding_dim)
        self.time_ln = nn.LayerNorm(time_embedding_dim)

        # Classification head:
        # Flattened conv output dimension:
        conv_output_dim = 64 * (latent_size // 8) * (latent_size // 8)
        # Concatenate flattened conv features with time embedding:
        self.fc = nn.Linear(conv_output_dim + time_embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

        # Segmentation head:
        # We condition the segmentation branch by adding a per-channel bias derived from time.
        self.seg_time_fc = nn.Linear(time_embedding_dim, 64)
        self.mask_conv = nn.Conv2d(64, out_dim, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

    def forward(self, x, timestep):
        """
        Args:
            x: Latents of shape (B, latent_channels, latent_size, latent_size)
            timestep: a tensor of shape (B,) (or scalar) representing the current diffusion timestep.
        Returns:
            logits_cls:  (B, out_dim) for multi-label classification.
            logits_mask: (B, out_dim, latent_size, latent_size) per-pixel location logits.
        """
        # Extract convolutional features
        x_conv = F.relu(self.conv1(x))
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = F.relu(self.conv3(x_conv))  # shape: (B, 64, latent_size/8, latent_size/8)

        # Get time embedding
        t_emb = self.time_emb(timestep)  # shape: (B, time_embedding_dim)
        t_emb = self.time_ln(t_emb)

        # Segmentation branch:
        # Project time embedding to a bias for conv channels.
        seg_time = self.seg_time_fc(t_emb)  # shape: (B, 64)
        seg_time = seg_time.unsqueeze(-1).unsqueeze(-1)  # shape: (B, 64, 1, 1)
        x_seg = x_conv + seg_time  # condition conv features with time
        mask_logits_8x8 = self.mask_conv(x_seg)  # shape: (B, out_dim, latent_size/8, latent_size/8)
        logits_mask = self.upsample(mask_logits_8x8)  # shape: (B, out_dim, latent_size, latent_size)

        # Classification branch:
        x_flat = x_conv.view(x_conv.size(0), -1)  # flatten conv features
        # Concatenate with time embedding:
        x_cat = torch.cat([x_flat, t_emb], dim=1)  # shape: (B, conv_output_dim + time_embedding_dim)
        x_fc = F.relu(self.fc(x_cat))
        logits_cls = self.out(x_fc)

        return logits_cls, logits_mask

class ConceptClassifierSegmenterWithTimeFiLM(nn.Module):
    """
    A multi-label classification + per-concept mask network,
    with FiLM-based time conditioning for both heads.
    """
    def __init__(
        self,
        latent_channels=4,
        latent_size=64,
        out_dim=8,
        hidden_dim=256,
        time_embedding_dim=32
    ):
        """
        Args:
            latent_channels: number of channels in the latents (often 4 for SD).
            latent_size: height/width of the latent (often 64 for SD).
            out_dim: number of concepts (e.g. <asset0>..).
            hidden_dim: hidden dimension for the classification MLP.
            time_embedding_dim: dimension of the sinusoidal time embedding.
        """
        super().__init__()

        # 1) Basic conv feature extractor
        self.conv1 = nn.Conv2d(latent_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # => final shape ~ (B,64,latent_size/8,latent_size/8)

        # 2) Time embedding
        self.time_emb = TimeEmbedding(time_embedding_dim)
        self.time_ln = nn.LayerNorm(time_embedding_dim)

        # 3) FiLM for segmentation
        # We produce 2 * 64 => gamma_seg, beta_seg, each shape (B,64)
        self.seg_film = nn.Linear(time_embedding_dim, 2 * 64)

        # Seg head => conv => upsample
        self.mask_conv = nn.Conv2d(64, out_dim, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        # 4) FiLM for classification
        # Flattened conv output dimension:
        self.conv_output_dim = 64 * (latent_size // 8) * (latent_size // 8)
        # We'll produce gamma_cls, beta_cls => each shape (B, conv_output_dim)
        self.cls_film = nn.Linear(time_embedding_dim, 2 * self.conv_output_dim)

        #TODO: May need to add a MLP (utilizing the hidden dim)
        # Then a final linear MLP for classification
        self.cls_out = nn.Linear(self.conv_output_dim, out_dim)

        # Initialize seg_film and cls_film to produce near-zero gamma/beta
        nn.init.zeros_(self.seg_film.weight)
        nn.init.zeros_(self.seg_film.bias)
        nn.init.zeros_(self.cls_film.weight)
        nn.init.zeros_(self.cls_film.bias)

    def forward(self, x, timestep):
        """
        Args:
            x:  (B, latent_channels, latent_size, latent_size)
            timestep: (B,) or scalar => current diffusion step
        Returns:
            logits_cls:  (B, out_dim) multi-label classification
            logits_mask: (B, out_dim, latent_size, latent_size) masks
        """
        # 1) Convolutional features
        x_conv = F.relu(self.conv1(x))
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = F.relu(self.conv3(x_conv))  # => shape (B,64,latent_size/8,latent_size/8)

        B, C, H, W = x_conv.shape

        # 2) Time embedding
        t_emb = self.time_emb(timestep)      # (B, time_embedding_dim)
        t_emb = self.time_ln(t_emb)          # stable scaling

        # 3) FiLM for segmentation
        film_seg = self.seg_film(t_emb)      # => (B, 128), chunk => gamma_seg, beta_seg
        gamma_seg, beta_seg = torch.chunk(film_seg, chunks=2, dim=1)  # each (B,64)
        gamma_seg = gamma_seg.unsqueeze(-1).unsqueeze(-1)  # (B,64,1,1)
        beta_seg  = beta_seg.unsqueeze(-1).unsqueeze(-1)

        # Apply scale+shift => x_seg = x_conv * (1 + gamma) + beta
        x_seg = x_conv * (1 + gamma_seg) + beta_seg
        mask_logits_8x8 = self.mask_conv(x_seg)  # => (B,out_dim,H,W)
        logits_mask = self.upsample(mask_logits_8x8)  # => (B,out_dim,latent_size,latent_size)

        # 4) FiLM for classification
        # Flatten conv features
        x_flat = x_conv.view(B, -1)  # => (B,conv_output_dim)

        film_cls = self.cls_film(t_emb)  # => (B, 2*conv_output_dim)
        gamma_cls, beta_cls = torch.chunk(film_cls, chunks=2, dim=1)  # => (B,conv_output_dim) each

        # x_film = x_flat * (1 + gamma_cls) + beta_cls
        x_film = x_flat * (1 + gamma_cls) + beta_cls

        # final classification => shape (B, out_dim)
        logits_cls = self.cls_out(x_film)

        return logits_cls, logits_mask

### Disgarded
class ConceptRegressor(nn.Module):
    """
    A simple regressor to predict the actual concept embedding vectors for each image.
    Typically you'd compare with MSE if you store ground-truth embeddings.
    """
    def __init__(self, latent_channels=4, latent_size=64, embedding_dim=768, hidden_dim=256):
        super().__init__()
        """
        Args:
            latent_channels: Number of channels in the latents.
            latent_size: Height/width of the latent.
            embedding_dim: Dimensionality of each concept embedding to predict.
            hidden_dim: Hidden dimension for the fully connected layer(s).
        """
        self.conv1 = nn.Conv2d(latent_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.fc = nn.Linear(64 * (latent_size // 8) * (latent_size // 8), hidden_dim)
        self.out = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Latents of shape (B, latent_channels, latent_size, latent_size)
        Returns:
            A predicted embedding of shape (B, embedding_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        embedding = self.out(x)
        return embedding