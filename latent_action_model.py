import torch
import torch.nn as nn
import torch.nn.functional as F

def load_latent_action_model(model_path, device):
    model = LatentActionVQVAE()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Intelligently find the state dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # If no specific key, assume the dict itself is the state_dict
            state_dict = checkpoint
    else:
        # If not a dict, assume the loaded object is the state_dict
        state_dict = checkpoint

    # Fix state dict keys by removing the '*orig_mod.' prefix for compiled models
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            fixed_state_dict[k.replace('_orig_mod.', '')] = v
        else:
            fixed_state_dict[k] = v
            
    model.load_state_dict(fixed_state_dict)
    
    # Gracefully get step/epoch number
    step = 0
    if isinstance(checkpoint, dict):
        step = checkpoint.get('step', checkpoint.get('epoch', 0))
        
    return model, step

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
    Layer 3: (21, 21) -> (10, 10)
    Layer 4: (10, 10) -> (5, 5)
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
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # z: (B, C, H, W)
        # Flatten spatial dimensions for vector quantization
        z_flat = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        z_flat = z_flat.view(-1, self.embedding_dim)  # (B*H*W, C)
        
        # Compute L2 distance to codebook
        d = (z_flat.pow(2).sum(1, keepdim=True)
             - 2 * z_flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))
        
        # Get nearest codebook indices
        encoding_indices = torch.argmin(d, dim=1)
        
        # Get quantized vectors
        quantized = self.embedding(encoding_indices)  # (B*H*W, C)
        
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
    def __init__(self, codebook_size=256, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(out_dim=embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(in_channels=embedding_dim)

    def forward(self, frame_t, frame_tp1, return_latent=False):
        # Input frames are already in (B, C, 84, 84) format
        # Concatenate along channel dimension (dim=1)
        x = torch.cat([frame_t, frame_tp1], dim=1)  # (B, 2*C, 84, 84)
        
        z = self.encoder(x)  # (B, 128, 5, 5)
        quantized, indices, commitment_loss, codebook_loss = self.quantizer(z)
        
        # Decode using current frame as conditioning
        recon = self.decoder(quantized, frame_t)
        
        if return_latent:
            return recon, indices, commitment_loss, codebook_loss, z
        else:
            return recon, indices, commitment_loss, codebook_loss

class ActionToLatentMLP(nn.Module):
    """
    A simple MLP that maps a one-hot encoded action to a sequence of latent codes.

    This model is state-less; it does not consider the current game observation.
    It learns the most probable latent transition for a given action.
    """
    def __init__(self, n_actions=4, latent_seq_len=25, latent_vocab_size=256, dropout_p=0.2):
        """
        Args:
            n_actions (int): The number of possible actions (size of one-hot vector).
            latent_seq_len (int): The length of the flattened latent code sequence (e.g., 5*5=25).
            latent_vocab_size (int): The size of the VQ-VAE codebook (e.g., 256).
            dropout_p (float): Dropout probability.
        """
        super().__init__()
        self.latent_seq_len = latent_seq_len
        self.latent_vocab_size = latent_vocab_size
        
        self.net = nn.Sequential(
            nn.Linear(n_actions, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            # The output layer produces logits for each of the 25 positions in the sequence.
            nn.Linear(256, latent_seq_len * latent_vocab_size)
        )

    def forward(self, action_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            action_one_hot: A batch of one-hot encoded actions. Shape: (B, n_actions)
        Returns:
            logits: The output logits. Shape: (B, latent_seq_len, latent_vocab_size)
        """
        logits = self.net(action_one_hot)
        # Reshape the output to be (Batch, Sequence Length, Vocabulary Size)
        # This is the format expected by the CrossEntropyLoss when comparing with the target.
        return logits.view(-1, self.latent_seq_len, self.latent_vocab_size)

    def sample(self, action_one_hot: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Autoregressively sample a latent code sequence for a given action.
        Note: This MLP is not autoregressive, so it predicts all positions at once.
              "Sampling" here means sampling from the predicted probability distribution
              for each position independently.

        Args:
            action_one_hot: A batch of one-hot encoded actions. Shape: (B, n_actions)
            temperature: A factor to control the randomness of sampling.
                         Higher temperature -> more random. Lower -> more greedy.
        
        Returns:
            sampled_indices: A sequence of sampled latent indices. Shape: (B, latent_seq_len)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(action_one_hot)
            
            # Apply temperature to the logits
            if temperature <= 0:
                # Greedy decoding for temperature=0 or less
                return logits.argmax(dim=-1)
            
            scaled_logits = logits / temperature
            
            # Convert logits to probabilities for each position in the sequence
            # Shape: (B, latent_seq_len, latent_vocab_size)
            probs = F.softmax(scaled_logits, dim=-1)
            
            # For each position in the sequence, sample one index from its distribution.
            # We need to reshape for torch.multinomial, which expects 2D input.
            # (B * latent_seq_len, latent_vocab_size)
            reshaped_probs = probs.view(-1, self.latent_vocab_size)
            
            # Sample 1 index for each of the (B * seq_len) distributions
            sampled_indices = torch.multinomial(reshaped_probs, num_samples=1)
            
            # Reshape back to the desired output shape
            return sampled_indices.view(-1, self.latent_seq_len)

class ActionStateToLatentMLP(nn.Module):
    """
    An MLP that maps a state (two consecutive frames) and a one-hot encoded action
    to a sequence of latent codes. This is a more powerful, stateful version of
    the ActionToLatentMLP.
    """
    def __init__(self, n_actions=4, latent_seq_len=25, codebook_size=256, frame_embedding_dim=128, hidden_dim=512, dropout_p=0.2):
        super().__init__()
        self.latent_seq_len = latent_seq_len
        self.codebook_size = codebook_size
        
        # Frame encoder for 2 RGB frames (6 channels, 84x84)
        # Follows a standard CNN architecture for feature extraction
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=8, stride=4, padding=0), # (B, 32, 20, 20)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), # (B, 64, 9, 9)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # (B, 64, 7, 7)
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, frame_embedding_dim),
            nn.ReLU(inplace=True),
        )
        
        # Combined MLP for action + frame features
        self.net = nn.Sequential(
            nn.Linear(n_actions + frame_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, latent_seq_len * codebook_size)
        )

    def forward(self, action_one_hot: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_one_hot: (B, n_actions)
            frames: (B, 6, 84, 84) -- two consecutive RGB frames
        Returns:
            logits: (B, latent_seq_len, codebook_size)
        """
        frame_features = self.frame_encoder(frames)
        combined = torch.cat([action_one_hot, frame_features], dim=1)
        logits = self.net(combined)
        return logits.view(-1, self.latent_seq_len, self.codebook_size)

    def sample(self, action_one_hot: torch.Tensor, frames: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample a latent code sequence for a given action and state.
        
        Args:
            action_one_hot: A batch of one-hot encoded actions. Shape: (B, n_actions)
            frames: A batch of two consecutive frames. Shape: (B, 6, 84, 84)
            temperature: A factor to control the randomness of sampling.
        
        Returns:
            sampled_indices: A sequence of sampled latent indices. Shape: (B, latent_seq_len)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(action_one_hot, frames)
            
            if temperature <= 0:
                return logits.argmax(dim=-1)
            
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            reshaped_probs = probs.view(-1, self.codebook_size)
            sampled_indices = torch.multinomial(reshaped_probs, num_samples=1)
            
            return sampled_indices.view(-1, self.latent_seq_len)