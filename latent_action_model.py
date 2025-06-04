import torch
import torch.nn as nn
import torch.nn.functional as F

def load_latent_action_model(model_path, device):
    model = LatentActionVQVAE()
    checkpoint = torch.load(model_path, map_location=device)
    # Fix state dict keys by removing the '*orig*mod.' prefix
    fixed_state_dict = {}
    for k, v in checkpoint['model'].items():
        if k.startswith('_orig_mod.'):
            fixed_state_dict[k.replace('_orig_mod.', '')] = v
        else:
            fixed_state_dict[k] = v
    model.load_state_dict(fixed_state_dict)
    return model, checkpoint['step']

class Encoder(nn.Module):
    """
    Encoder for VQ-VAE latent action model.
    - Input: Concatenated current and next frames (B, 6, 84, 84) for RGB
    - Output: Latent feature map (B, 128, 5, 5)

    The architecture uses 4 Conv2d layers with stride=2 to downsample the input.
    The kernel sizes and paddings are chosen to ensure the final output spatial size is exactly (5, 5) for input (84, 84).

    Downsampling calculation for (H, W) = (84, 84):
    Layer 1: (84, 84) -> (42, 42)
    Layer 2: (42, 42) -> (21, 21)
    Layer 3: (21, 21) -> (11, 11)
    Layer 4: (11, 11) -> (5, 5)
    """
    def __init__(self, in_channels=6, hidden_dims=[64, 128, 256, 512], out_dim=128):
        super().__init__()
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(hidden_dims):
            # Standard downsampling: kernel=4, stride=2, padding=1
            # This halves the spatial size each time
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        # Project to latent embedding dimension (128)
        self.project = nn.Conv2d(hidden_dims[-1], out_dim, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.project(x)
        return x  # (B, 128, 5, 5)

class VectorQuantizer(nn.Module):
    """
    Vector quantization layer for VQ-VAE.
    - Codebook size: 256
    - Embedding dim: 128
    - Uses straight-through estimator for backprop.
    - Returns quantized latents, indices, and losses.
    """
    def __init__(self, num_embeddings=256, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # z: (B, C, H, W)
        # Flatten spatial dimensions for vector quantization
        z_flat = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        z_flat = z_flat.view(-1, self.embedding_dim)  # (B*H*W, C)
        
        # Compute L2 distance to codebook
        d = (z_flat.pow(2).sum(1, keepdim=True)
             - 2 * z_flat @ self.embeddings.weight.t()
             + self.embeddings.weight.pow(2).sum(1))
        
        # Get nearest codebook indices
        encoding_indices = torch.argmin(d, dim=1)
        
        # Get quantized vectors
        quantized = self.embeddings(encoding_indices)  # (B*H*W, C)
        
        # Reshape back to original dimensions
        quantized = quantized.view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)  # (B, H, W, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        # Compute losses
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)
        codebook_loss = F.mse_loss(quantized, z.detach())
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        return quantized, encoding_indices.view(z.shape[0], z.shape[2], z.shape[3]), commitment_loss, codebook_loss

class Decoder(nn.Module):
    """
    Decoder for VQ-VAE latent action model.
    - Input: Quantized latent (B, 128, 5, 5) and current frame (B, 3, 84, 84)
    - Output: Reconstructed next frame (B, 3, 84, 84)

    The decoder upsamples the latent representation back to the original frame size using 4 transposed conv layers.
    """
    def __init__(self, in_channels=128, cond_channels=3, hidden_dims=[512, 256, 128, 64], out_channels=3):
        super().__init__()
        # Process current frame for conditioning
        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Combine latent and conditioning
        self.fc = nn.Conv2d(in_channels+128, hidden_dims[0], kernel_size=1)
        up_layers = []
        c_in = hidden_dims[0]
        for c_out in hidden_dims[1:]:
            # Standard upsampling: kernel=4, stride=2, padding=1
            up_layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            up_layers.append(nn.BatchNorm2d(c_out))
            up_layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        # Final upsampling to get to (84, 84)
        up_layers.append(nn.ConvTranspose2d(c_in, out_channels, kernel_size=4, stride=2, padding=1))
        self.up = nn.Sequential(*up_layers)
    
    def forward(self, z, cond):
        # Process conditioning frame
        cond_feat = self.cond_conv(cond)
        # Resize conditioning to match latent spatial size
        cond_feat = F.interpolate(cond_feat, size=z.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate and upsample
        x = torch.cat([z, cond_feat], dim=1)
        x = self.fc(x)
        x = self.up(x)
        # Guarantee output is (B, 3, 84, 84) by resizing
        x = F.interpolate(x, size=(84, 84), mode='bilinear', align_corners=False)
        return x

class LatentActionVQVAE(nn.Module):
    """
    Full VQ-VAE model for latent action prediction.
    - Encoder: Extracts latent from (frame_t, frame_t+1)
    - VectorQuantizer: Discretizes latent
    - Decoder: Reconstructs next frame from quantized latent and current frame
    """
    def __init__(self, codebook_size=256, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(out_dim=embedding_dim)
        self.vq = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(in_channels=embedding_dim)

    def forward(self, frame_t, frame_tp1, return_latent=False):
        # Input frames are already in (B, C, 84, 84) format
        # Concatenate along channel dimension (dim=1)
        x = torch.cat([frame_t, frame_tp1], dim=1)  # (B, 2*C, 84, 84)
        
        z = self.encoder(x)  # (B, 128, 5, 5)
        quantized, indices, commitment_loss, codebook_loss = self.vq(z)
        
        # Decode using current frame as conditioning
        recon = self.decoder(quantized, frame_t)
        
        if return_latent:
            return recon, indices, commitment_loss, codebook_loss, z
        else:
            return recon, indices, commitment_loss, codebook_loss

class ActionToLatentMLP(nn.Module):
    def __init__(self, input_dim=4, hidden1=512, hidden2=256, latent_dim=35, codebook_size=256, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, latent_dim * codebook_size)
        )

    def forward(self, x):
        out = self.net(x)  # (batch, latent_dim * codebook_size)
        out = out.view(-1, self.latent_dim, self.codebook_size)
        return out

    def sample_latents(self, logits, temperature=1.0):
        # logits: (batch, 35, 256)
        if temperature <= 0:
            raise ValueError("Temperature must be > 0")
        probs = F.softmax(logits / temperature, dim=-1)  # (batch, 35, 256)
        batch, latent_dim, codebook_size = probs.shape
        # Sample for each position
        samples = torch.multinomial(probs.view(-1, codebook_size), 1).view(batch, latent_dim)
        return samples

class ActionStateToLatentMLP(nn.Module):
    def __init__(self, action_dim=4, hidden1=512, hidden2=256, latent_dim=35, codebook_size=256, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        # Frame encoder for 2 RGB frames (6 channels, 210x160)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=8, stride=4),  # (B, 16, 51, 39)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # (B, 32, 24, 18)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (B, 64, 11, 8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 8, 128),
            nn.ReLU(),
        )
        # Combined MLP for action + frame features
        self.net = nn.Sequential(
            nn.Linear(action_dim + 128, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, latent_dim * codebook_size)
        )

    def forward(self, action, frames):
        # action: (B, 4), frames: (B, 6, 210, 160)
        frame_features = self.frame_encoder(frames)
        combined = torch.cat([action, frame_features], dim=1)
        out = self.net(combined)
        return out.view(-1, self.latent_dim, self.codebook_size)

    def sample_latents(self, logits, temperature=1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be > 0")
        probs = F.softmax(logits / temperature, dim=-1)
        batch, latent_dim, codebook_size = probs.shape
        samples = torch.multinomial(probs.view(-1, codebook_size), 1).view(batch, latent_dim)
        return samples