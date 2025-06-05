"""
VQ-VAE model for latent action prediction in Atari games, based on the provided reference architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder for VQ-VAE latent action model.
    - Input: Concatenated current and next frames (B, 2*C, H, W), e.g., (B, 6, 160, 210) for RGB.
    - Output: Latent feature map (B, embedding_dim, 5, 7).
    """
    def __init__(self, in_channels=6, hidden_dims=None, out_dim=128, dropout_p=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 512]
        
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(hidden_dims):
            if i < 4:
                # Standard downsampling: kernel=4, stride=2, padding=1. Halves spatial size.
                layers.append(nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            else:
                # Last layer: custom kernel for width to get (5, 7) output from (10, 14) input.
                layers.append(nn.Conv2d(c_in, c_out, kernel_size=(4,7), stride=2, padding=(1,3)))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_p))
            c_in = c_out
            
        self.conv = nn.Sequential(*layers)
        self.project = nn.Conv2d(hidden_dims[-1], out_dim, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.project(x)
        return x  # (B, 128, 5, 7)

class VectorQuantizer(nn.Module):
    """
    Vector quantization layer.
    - Discretizes the continuous output of the encoder.
    - Uses straight-through estimator for backpropagation.
    - Calculates VQ loss and codebook usage (perplexity).
    """
    def __init__(self, num_embeddings=256, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        # z: (B, C, H, W) -> (B, H, W, C)
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_permuted.view(-1, self.embedding_dim)

        # Calculate distances to codebook vectors
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        
        # Find closest encodings
        min_encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(min_encoding_indices).view(z_permuted.shape)
        
        # Calculate losses
        codebook_loss = F.mse_loss(quantized, z_permuted.detach())
        scaled_commitment_loss = self.commitment_cost * F.mse_loss(z_permuted, quantized.detach())
        vq_loss_total = codebook_loss + scaled_commitment_loss

        # Straight-through estimator
        quantized_for_decoder = z + (quantized.permute(0, 3, 1, 2).contiguous() - z).detach()

        # Calculate perplexity (codebook usage)
        encodings = F.one_hot(min_encoding_indices, self.num_embeddings).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (
            quantized_for_decoder,
            vq_loss_total,
            codebook_loss,
            scaled_commitment_loss,
            perplexity,
            min_encoding_indices,
        )

class Decoder(nn.Module):
    """
    Decoder for VQ-VAE latent action model.
    - Input: Quantized latent (B, 128, 5, 7) and current frame (B, 3, 160, 210).
    - Output: Reconstructed next frame (B, 3, 160, 210).
    - Conditioning: The current frame is processed and concatenated with the latent vector.
    """
    def __init__(self, in_channels=128, cond_channels=3, hidden_dims=None, out_channels=3, dropout_p=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64] # Note: one less than encoder as first is handled by FC
            
        # Process current frame for conditioning
        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Combine latent and conditioning features
        self.fc = nn.Conv2d(in_channels + 128, hidden_dims[0], kernel_size=1)
        
        up_layers = []
        c_in = hidden_dims[0]
        # Start from the second hidden_dim since the first is the output of the FC layer
        for c_out in hidden_dims[1:]:
            up_layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            up_layers.append(nn.BatchNorm2d(c_out))
            up_layers.append(nn.ReLU(inplace=True))
            up_layers.append(nn.Dropout(dropout_p))
            c_in = c_out
            
        # Final upsampling layer to get to the target size
        up_layers.append(nn.ConvTranspose2d(c_in, out_channels, kernel_size=4, stride=2, padding=1))
        self.up = nn.Sequential(*up_layers)

    def forward(self, z, cond):
        # Process conditioning frame
        cond_feat = self.cond_conv(cond)
        
        # Resize conditioning features to match latent spatial size
        cond_feat = F.interpolate(cond_feat, size=z.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate and upsample
        x = torch.cat([z, cond_feat], dim=1)
        x = self.fc(x)
        x = self.up(x)
        
        # Guarantee output is the same size as the conditioning frame (e.g., 160, 210)
        x = F.interpolate(x, size=cond.shape[2:], mode='bilinear', align_corners=False)
        return x

class VQVAE(nn.Module):
    """
    Full VQ-VAE model for latent action prediction.
    """
    def __init__(self, input_channels_per_frame=3, embedding_dim=128, num_embeddings=256, commitment_cost=0.25, dropout_p=0.1):
        super().__init__()
        # The encoder takes two frames concatenated, so 2 * channels
        self.encoder = Encoder(
            in_channels=2 * input_channels_per_frame, 
            out_dim=embedding_dim,
            dropout_p=dropout_p
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost
        )
        # The decoder takes the latent and one frame for conditioning
        self.decoder = Decoder(
            in_channels=embedding_dim, 
            cond_channels=input_channels_per_frame, 
            out_channels=input_channels_per_frame,
            dropout_p=dropout_p
        )

    def forward(self, frame_t, frame_tp1):
        # The reference model's convolutions expect (H, W) of (160, 210).
        # Our dataset provides (210, 160). We permute H and W.
        frame_t_permuted = frame_t.permute(0, 1, 3, 2)
        frame_tp1_permuted = frame_tp1.permute(0, 1, 3, 2)
        
        # Concatenate frames to find the latent "action"
        x = torch.cat([frame_t_permuted, frame_tp1_permuted], dim=1)
        
        latents_e = self.encoder(x)
        
        (
            quantized_for_decoder,
            vq_loss_total,
            codebook_loss,
            scaled_commitment_loss,
            perplexity,
            min_encoding_indices,
        ) = self.quantizer(latents_e)
        
        # Reconstruct frame_tp1 from latent and frame_t
        reconstructed_frame_tp1_permuted = self.decoder(quantized_for_decoder, frame_t_permuted)
        
        # Permute back to the original data format (C, 210, 160)
        reconstructed_frame_tp1 = reconstructed_frame_tp1_permuted.permute(0, 1, 3, 2)
        
        return (
            reconstructed_frame_tp1,
            vq_loss_total,
            codebook_loss,
            scaled_commitment_loss,
            perplexity,
            latents_e,
            min_encoding_indices,
            quantized_for_decoder # for debug loss
        )

    def calculate_loss(self,
                       frame_tp1_original,
                       reconstructed_frame_tp1,
                       vq_loss_total_from_quantizer,
                       min_encoding_indices,
                       codebook_entropy_reg_weight,
                       quantized_for_decoder_debug,
                       debug_embedding_weight,
                       use_aux_debug_loss,
                       **kwargs): # Absorb unused loss components for compatibility
        
        reconstruction_loss = F.mse_loss(reconstructed_frame_tp1, frame_tp1_original)
        
        # --- Optional: Codebook Entropy Regularization ---
        entropy_reg_term = 0.0
        if codebook_entropy_reg_weight > 0.0 and min_encoding_indices is not None:
            # Calculate entropy of codebook usage for the current batch
            encodings = F.one_hot(min_encoding_indices, self.quantizer.num_embeddings).float()
            avg_probs = encodings.mean(dim=0)
            # We want to maximize entropy, which is equivalent to minimizing negative entropy.
            entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
            # The regularization term encourages higher entropy (more uniform usage)
            entropy_reg_term = -codebook_entropy_reg_weight * entropy

        # --- Optional: Auxiliary Loss for Debugging ---
        aux_debug_loss = torch.tensor(0.0, device=reconstructed_frame_tp1.device)
        if use_aux_debug_loss and debug_embedding_weight is not None:
            # This loss encourages embeddings to stay close to zero, preventing them from growing too large.
            # It's a form of weight decay, applied only to the VQ embeddings.
            aux_debug_loss = torch.mean(debug_embedding_weight.pow(2))

        total_loss = reconstruction_loss + vq_loss_total_from_quantizer + entropy_reg_term + aux_debug_loss
        
        return total_loss, reconstruction_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), aux_debug_loss, entropy_reg_term

# Next: VQVAE main model

# Further components (Encoder, Decoder, VQVAE main model) will be added below. 