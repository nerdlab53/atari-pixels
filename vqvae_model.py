"""VQ-VAE Model Components (VectorQuantizer, Encoder, Decoder, VQVAE)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer.

    Args:
        num_embeddings (int): Number of vectors in the codebook (K).
        embedding_dim (int): Dimensionality of each vector in the codebook (D).
        commitment_cost (float): Weight for the commitment loss component.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize the codebook embeddings. These are the learnable vectors.
        # Shape: (K, D)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialize weights with a uniform distribution, common practice for VQ-VAEs.
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, latents_e):
        """
        Forward pass of the VQ layer.

        Args:
            latents_e (Tensor): Output of the encoder. 
                               Expected shape: (Batch, Channels, Height, Width)
                               where Channels is the embedding_dim.

        Returns:
            quantized_latents (Tensor): Quantized latent vectors.
                                        Shape: (Batch, Channels, Height, Width)
            loss (Tensor): The VQ loss (commitment loss).
            perplexity (Tensor): A measure of codebook usage.
            min_encodings (Tensor): The flat tensor of chosen codebook indices.
        """
        # Reshape encoder output: (B, C, H, W) -> (B*H*W, C)
        # C (Channels) should be equal to self.embedding_dim
        assert latents_e.shape[1] == self.embedding_dim, \
            f"Input channel dimension ({latents_e.shape[1]}) must match embedding_dim ({self.embedding_dim})"
        
        flat_latents_e = latents_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances between encoder outputs and codebook vectors
        # distances = sum((z_e - z_q)^2) = z_e^2 + z_q^2 - 2*z_e*z_q
        # z_e^2: (B*H*W, 1)
        sum_sq_latents_e = torch.sum(flat_latents_e**2, dim=1, keepdim=True)
        # z_q^2: (1, K)
        sum_sq_embeddings = torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
        # -2*z_e*z_q: (B*H*W, K)
        term_multiply = torch.matmul(flat_latents_e, self.embedding.weight.t())
        
        distances = sum_sq_latents_e + sum_sq_embeddings - 2.0 * term_multiply
        
        # Find the closest codebook vector (indices)
        # min_encoding_indices shape: (B*H*W,)
        min_encoding_indices = torch.argmin(distances, dim=1)
        
        # Get the quantized latent vectors using the indices
        # self.embedding.weight shape (K, D)
        # quantized_flat shape: (B*H*W, D)
        quantized_flat = self.embedding(min_encoding_indices)
        
        # Reshape quantized vectors back to original latent space dimensions
        # (B*H*W, D) -> (B, H, W, D) -> (B, D, H, W)
        quantized_latents = quantized_flat.view_as(latents_e.permute(0, 2, 3, 1).contiguous())
        quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous()

        # --- Calculate VQ Loss ---
        # 1. Commitment Loss: Encourages the encoder output to be close to the chosen codebook vector.
        # Use detached (stopped gradient) quantized latents for the encoder's target.
        commitment_loss = F.mse_loss(latents_e, quantized_latents.detach())
        
        # 2. Codebook Loss (part of the original VQ-VAE formulation, sometimes called embedding loss):
        # Encourages the codebook vectors to be close to the encoder outputs that map to them.
        # This loss is applied to the codebook; encoder gradients don't flow back from this. 
        # Modern implementations often only use commitment_loss for the VQ part and let the main 
        # reconstruction loss update the codebook via the straight-through estimator.
        # For simplicity and following common practice, we will primarily focus on commitment loss here 
        # as the VQ-specific loss. The embeddings are updated via STE from the main model loss.
        # codebook_loss = F.mse_loss(quantized_latents, latents_e.detach())
        # vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Simplified VQ loss: only the commitment part scaled by commitment_cost.
        # The embeddings themselves are updated by the gradient from the decoder, passed
        # through the straight-through estimator.
        vq_loss = self.commitment_cost * commitment_loss

        # Straight-Through Estimator (STE)
        # In the backward pass, the gradient from `quantized_latents` is copied to `latents_e`.
        # This allows the encoder to receive gradients as if it produced the quantized values directly.
        quantized_latents = latents_e + (quantized_latents - latents_e).detach()

        # --- Calculate Perplexity (a measure of codebook usage) ---
        # This is for monitoring, not part of the loss that's optimized directly here.
        # It indicates how many codebook vectors are being effectively used.
        # Higher perplexity is generally better, suggesting richer codebook usage.
        avg_probs = torch.mean(torch.eye(self.num_embeddings, device=latents_e.device)[min_encoding_indices], dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized_latents, vq_loss, perplexity, min_encoding_indices.view(latents_e.shape[0], -1)

class Encoder(nn.Module):
    """
    Encoder network for the VQ-VAE.
    Takes frame_t and frame_t+1, computes their difference, and concatenates all three.
    Downsamples the combined input to a latent representation.

    Args:
        input_channels (int): Number of channels in each input frame (e.g., 3 for RGB, 1 for grayscale).
        embedding_dim (int): Dimensionality of the latent embeddings from the VQ layer.
                             The encoder's final 1x1 conv will output this many channels.
    """
    def __init__(self, input_channels_per_frame: int, embedding_dim: int):
        super().__init__()
        self.input_channels_per_frame = input_channels_per_frame
        
        # --- START: MODIFICATION FOR EXPLICIT DIFFERENCE INPUT ---
        # The actual input to the first conv layer will be 3 times the channels of a single frame
        # (frame_t, frame_t+1, diff_frame)
        conv_input_channels = input_channels_per_frame * 3
        # --- END: MODIFICATION FOR EXPLICIT DIFFERENCE INPUT ---

        # Channel progression based on the plan, adjusted for the new conv_input_channels
        # Original plan progression: 6->64->128->256->512->512. If input_channels_per_frame=3 (RGB), conv_input_channels=9.
        # If input_channels_per_frame=1 (Grayscale), conv_input_channels=3.
        # We need to map conv_input_channels to 64, then proceed.

        self.layers = nn.Sequential(
            # Layer 1: (B, conv_input_channels, 160, 210) -> (B, 64, 80, 105)
            nn.Conv2d(conv_input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 2: (B, 64, 80, 105) -> (B, 128, 40, 52) (W=105 -> (105-4+2)/2+1 = 52.5 -> 52 (floor))
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 3: (B, 128, 40, 52) -> (B, 256, 20, 26)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 4: (B, 256, 20, 26) -> (B, 512, 10, 13)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Layer 5: (B, 512, 10, 13) -> (B, 512, 5, 7)
            # To get H=5 from 10: kernel=4, padding=1 -> (10-4+2)/2+1 = 4. Needs K=2,P=0 or K=3,P=0.5(invalid) or K=4,P=1 works. (10-4+2)/2+1 = 4. Hmm. 
            # (H_in - K + 2P)/S + 1 = H_out => (10 - K + 2P)/2 + 1 = 5 => 10 - K + 2P = 8 => K - 2P = 2.
            # For W=7 from 13: (13 - K + 2P)/2 + 1 = 7 => 13 - K + 2P = 12 => K - 2P = 1.
            # This means K, P cannot be the same for H and W if K-2P is different.
            # Let's use K=3, P_h=0 for H: (10-3)/2+1 = 3.5+1=4. No. K=3,P_h=1: (10-3+2)/2+1 = 4.5+1=5.
            # Let's use K=3, P_w=1 for W: (13-3+2)/2+1 = 6+1=7.
            # So, kernel_size=3, stride=2, padding=1 works for both H and W to get 5x7.
            # H: (10 - 3 + 2*1)/2 + 1 = (9)/2 + 1 = 4.5 + 1 = 5 (floor) or 5.5 -> 5
            # W: (13 - 3 + 2*1)/2 + 1 = (12)/2 + 1 = 6 + 1 = 7
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Final 1x1 convolution to project to embedding_dim channels
            # Output: (B, embedding_dim, 5, 7)
            nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
            # No BN or ReLU after this layer, as its output goes to the VQ layer.
        )

    def forward(self, frame_t: torch.Tensor, frame_tp1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_t (Tensor): Batch of current frames (B, C_in, H, W).
            frame_tp1 (Tensor): Batch of next frames (B, C_in, H, W).

        Returns:
            latents_e (Tensor): Encoded latents (B, embedding_dim, H_latent, W_latent).
        """
        # Ensure frames are normalized to [0,1] if not already done by dataset
        # Assuming dataset handles normalization. If not, add frame_t/255.0 here.
        
        # --- START: MODIFICATION FOR EXPLICIT DIFFERENCE INPUT ---
        frame_diff = frame_tp1 - frame_t # Difference frame
        # Concatenate along the channel dimension
        # (B, C_in_per_frame*3, H, W)
        encoder_input = torch.cat((frame_t, frame_tp1, frame_diff), dim=1)
        # --- END: MODIFICATION FOR EXPLICIT DIFFERENCE INPUT ---
        
        latents_e = self.layers(encoder_input)
        return latents_e

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation Layer."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input feature map (B, C, H, W).
            gamma (Tensor): Scale factors (B, C, 1, 1) or (B, C).
            beta (Tensor): Shift factors (B, C, 1, 1) or (B, C).
        Returns:
            modulated_x (Tensor): Modulated feature map.
        """
        # Ensure gamma and beta are broadcastable to x
        if gamma.ndim == 2: # (B, C) -> (B, C, 1, 1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        if beta.ndim == 2: # (B, C) -> (B, C, 1, 1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        return gamma * x + beta

class Decoder(nn.Module):
    """
    Decoder network for the VQ-VAE.
    Takes quantized latents and frame_t (for FiLM conditioning) to reconstruct frame_t+1.

    Args:
        embedding_dim (int): Dimensionality of the quantized latent vectors.
        frame_t_channels (int): Number of channels in frame_t (e.g., 3 for RGB).
        output_channels (int): Number of channels in the reconstructed frame_t+1 (e.g., 3 for RGB).
    """
    def __init__(self, embedding_dim: int, frame_t_channels: int, output_channels: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.frame_t_channels = frame_t_channels
        self.output_channels = output_channels

        # --- START: FiLM CONDITIONING MODIFICATION ---
        # Conditioning network to process frame_t and generate FiLM parameters (gamma, beta)
        # This network will downsample frame_t to match spatial dimensions of some decoder layers
        # or produce global conditioning vectors.
        # Let's design it to produce gammas/betas for a couple of decoder layers.
        # Example: Modulate after the first and third upsampling layers in the decoder.
        # Decoder channel progression: 512 -> 512 -> 256 -> 128 -> 64 -> output_channels
        # We need to generate 2*512 params for first FiLM and 2*256 for second FiLM (if applied there)
        
        # Conditioning path for frame_t. This is a small CNN.
        # Input: frame_t (B, frame_t_channels, 160, 210)
        self.frame_t_conditioning_path = nn.Sequential(
            nn.Conv2d(frame_t_channels, 32, kernel_size=5, stride=2, padding=2), # (B, 32, 80, 105)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),              # (B, 64, 40, 53)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),             # (B, 128, 20, 27)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)) # Global average pooling to get (B, 128, 1, 1)
        )

        # Linear layers to generate FiLM parameters (gamma, beta) for decoder layers
        # Decoder layer 1 (output channels 512)
        self.film_params_dec1_channels = 512
        self.fc_film_dec1 = nn.Linear(128, self.film_params_dec1_channels * 2) # *2 for gamma and beta
        
        # Decoder layer 3 (output channels 256)
        self.film_params_dec3_channels = 256
        self.fc_film_dec3 = nn.Linear(128, self.film_params_dec3_channels * 2) # *2 for gamma and beta

        self.film_layer = FiLMLayer()
        # --- END: FiLM CONDITIONING MODIFICATION ---

        # Main decoder path using Transposed Convolutions
        # Input to first TConv: (B, embedding_dim, 5, 7)
        # Target channel progression: embedding_dim -> 512 -> 512 -> 256 -> 128 -> 64 -> output_channels
        
        self.decoder_layers = nn.ModuleList([
            # Layer 1: (B, embedding_dim, 5, 7) -> (B, 512, 10, 13) (Matches Encoder L4 output)
            # H: (5-1)*2 - 2*1 + 3 + 1 = 8 - 2 + 3 + 1 = 10
            # W: (7-1)*2 - 2*1 + 3 + 1 = 12 - 2 + 3 + 1 = 14. K=3,P=1,S=2 -> (7-1)*2 -2*1 + 3 +1 = 14.
            # We need to map to (10,13), so padding might need adjustment or output_padding
            # (H_in - 1)*S - 2P + K + OP = H_out
            # H: (5-1)*2 - 2*1 + K_h + OP_h = 10 => 8 - 2 + K_h + OP_h = 10 => K_h + OP_h = 4
            # W: (7-1)*2 - 2*1 + K_w + OP_w = 13 => 12 - 2 + K_w + OP_w = 13 => K_w + OP_w = 3
            # Using K=4, S=2, P=1 (as in encoder) but for TConv:
            # H_out = (H_in - 1)*S - 2*P + K = (5-1)*2 - 2*1 + 4 = 8-2+4 = 10.
            # W_out = (W_in - 1)*S - 2*P + K = (7-1)*2 - 2*1 + 4 = 12-2+4 = 14. Close.
            # Let's use K=4, S=2, P=1 for TConv. output_padding might be needed for exact W=13.
            nn.ConvTranspose2d(embedding_dim, 512, kernel_size=4, stride=2, padding=1), # Output (10, 14)
            # This layer will be FiLM conditioned.
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Layer 2: (B, 512, 10, 14) -> (B, 256, 20, 27) (Matches Encoder L3 output, W needs adjustment)
            # H: (10-1)*2 - 2*1 + 4 = 18-2+4 = 20.
            # W: (14-1)*2 - 2*1 + 4 = 26-2+4 = 28.
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # Output (20, 28)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # This layer will be FiLM conditioned.

            # Layer 3: (B, 256, 20, 28) -> (B, 128, 40, 53) (Matches Encoder L2 output, W needs adjustment)
            # H: (20-1)*2 - 2*1 + 4 = 38-2+4 = 40.
            # W: (28-1)*2 - 2*1 + 4 = 54-2+4 = 56.
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # Output (40, 56)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 4: (B, 128, 40, 56) -> (B, 64, 80, 105) (Matches Encoder L1 output, W needs adjustment)
            # H: (40-1)*2 - 2*1 + 4 = 78-2+4 = 80.
            # W: (56-1)*2 - 2*1 + 4 = 110-2+4 = 112.
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=(0,1)), # (B, 64, 80, 111+1=112) no... -> (B, 64, 80, 105), need W=105 from 52. S=2 P=1 K=4: (52-1)*2 -2*1 + 4 = 102-2+4 = 104. Add op=1 for 105.
            # For H_in=40, W_in=56. Want H_out=80, W_out=105.
            # H: (40-1)*2 - 2*1 + 4 = 78-2+4 = 80.
            # W: (56-1)*2 - 2*1 + 4 = 110-2+4 = 112. Need to get W_out=105.
            # To get 105 from 56: (56-1)*S -2P + K +OP = 105. If S=2, K=4, P=1: 110-2+4 = 112. op_w = -7 (not possible)
            # Let's adjust kernel for this layer for W. Target (80, 105) from (40, 56)
            # H: (40-1)*2+0-2*0+K_h=80 => 78+K_h=80 => K_h=2. S=2,P=0,K=2. Or S=2,P=1,K=4.
            # W: (56-1)*2+0-2*0+K_w=105 => 110+K_w=105 => K_w=-5. (Not possible with K=4,S=2,P=1)
            # The encoder's layer 2 output was (40, 52). If decoder input is (40,52), output (80, 104 or 105)
            # If K=4,S=2,P=1: W_out = (52-1)*2-2*1+4 = 102-2+4 = 104. Add output_padding=(0,1) to get 105.
            # So, if layer 3 output (40,52), layer 4 TConv K=4,S=2,P=1,OP=(0,1) -> (80,105)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 5: (B, 64, 80, 105) -> (B, output_channels, 160, 210)
            # H: (80-1)*2 - 2*1 + 4 = 158-2+4 = 160.
            # W: (105-1)*2 - 2*1 + 4 = 208-2+4 = 210.
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
            # Final layer, usually followed by a sigmoid to scale output to [0,1] or tanh to [-1,1]
            # If dataset ensures [0,1] input, and MSE loss, sigmoid is appropriate.
        ])
        
        # Latent_map (input to Decoder): (embedding_dim, 7, 5) (H_latent=7, W_latent=5)
        self.decoder_tconv_layers = nn.ModuleList([
            # Dec_L1: Input(emb_dim, 7, 5) -> Output(512, 13, 10)
            nn.ConvTranspose2d(embedding_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=(0,1)),
            # Dec_L2: Input(512, 13, 10) -> Output(256, 26, 20)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=(0,0)),
            # Dec_L3: Input(256, 26, 20) -> Output(128, 52, 40)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(0,0)),
            # Dec_L4: Input(128, 52, 40) -> Output(64, 105, 80)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=(1,0)),
            # Dec_L5: Input(64, 105, 80) -> Output(output_channels, 210, 160)
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1, output_padding=(0,0))
        ])
        
        self.decoder_bn_relu_blocks = nn.ModuleList([
            nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(inplace=True)),
            nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(inplace=True)),
            nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True)),
            # No BN/ReLU after the final TConv layer, it goes to sigmoid
        ])

    def forward(self, quantized_latents: torch.Tensor, frame_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            quantized_latents (Tensor): Quantized latent vectors from VQ layer (B, embedding_dim, H_latent, W_latent).
            frame_t (Tensor): Batch of current frames for FiLM conditioning (B, frame_t_channels, H_orig, W_orig).

        Returns:
            reconstructed_frame_tp1 (Tensor): Reconstructed next frame (B, output_channels, H_orig, W_orig).
        """
        # --- START: FiLM CONDITIONING MODIFICATION ---
        # Generate FiLM parameters from frame_t
        conditioning_features = self.frame_t_conditioning_path(frame_t) # (B, 128, 1, 1)
        conditioning_features = conditioning_features.view(conditioning_features.size(0), -1) # (B, 128)

        film_params_dec1 = self.fc_film_dec1(conditioning_features) # (B, film_params_dec1_channels * 2)
        gamma1, beta1 = torch.split(film_params_dec1, self.film_params_dec1_channels, dim=1)
        
        film_params_dec3 = self.fc_film_dec3(conditioning_features) # (B, film_params_dec3_channels * 2)
        # Note: Original plan was to FiLM layer 3 (output 256 channels), but decoder_tconv_layers[1] outputs 256 channels.
        # So we target the output of decoder_tconv_layers[1] (which is input to decoder_bn_relu_blocks[1])
        gamma3, beta3 = torch.split(film_params_dec3, self.film_params_dec3_channels, dim=1)
        # --- END: FiLM CONDITIONING MODIFICATION ---

        x = quantized_latents
        
        # Iterate through decoder layers, applying FiLM where specified
        x = self.decoder_tconv_layers[0](x)
        x = self.film_layer(x, gamma1, beta1) # FiLM after first TConv
        x = self.decoder_bn_relu_blocks[0](x)
        
        x = self.decoder_tconv_layers[1](x)
        x = self.film_layer(x, gamma3, beta3) # FiLM after second TConv (was planned for 3rd, maps to channels of this layer)
        x = self.decoder_bn_relu_blocks[1](x)
        
        x = self.decoder_tconv_layers[2](x)
        x = self.decoder_bn_relu_blocks[2](x)
        
        x = self.decoder_tconv_layers[3](x)
        x = self.decoder_bn_relu_blocks[3](x)
        
        x = self.decoder_tconv_layers[4](x)
        
        # Apply sigmoid to the final output to scale pixel values to [0, 1]
        reconstructed_frame_tp1 = torch.sigmoid(x)
        
        return reconstructed_frame_tp1

class VQVAE(nn.Module):
    """
    VQ-VAE model combining Encoder, VectorQuantizer, and Decoder.

    Args:
        input_channels_per_frame (int): Number of channels in each input frame (e.g., 3 for RGB).
        embedding_dim (int): Dimensionality of the latent embeddings.
        num_embeddings (int): Number of vectors in the codebook.
        commitment_cost (float): Weight for the commitment loss in the VQ layer.
        frame_t_channels_for_film (int): Number of channels in frame_t used for FiLM conditioning.
                                         This should be the same as input_channels_per_frame.
        output_channels_decoder (int): Number of output channels for the decoder (reconstructed frame_t+1).
                                       This should be the same as input_channels_per_frame.
    """
    def __init__(self, 
                 input_channels_per_frame: int, 
                 embedding_dim: int, 
                 num_embeddings: int, 
                 commitment_cost: float,
                 frame_t_channels_for_film: int, # Should be same as input_channels_per_frame
                 output_channels_decoder: int # Should be same as input_channels_per_frame
                 ):
        super().__init__()
        self.encoder = Encoder(input_channels_per_frame=input_channels_per_frame, 
                               embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, 
                                         embedding_dim=embedding_dim, 
                                         commitment_cost=commitment_cost)
        self.decoder = Decoder(embedding_dim=embedding_dim, 
                               frame_t_channels=frame_t_channels_for_film, 
                               output_channels=output_channels_decoder)

    def forward(self, frame_t: torch.Tensor, frame_tp1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VQ-VAE.

        Args:
            frame_t (Tensor): Batch of current frames (B, C, H, W).
            frame_tp1 (Tensor): Batch of next frames (B, C, H, W).

        Returns:
            reconstructed_frame_tp1 (Tensor): The decoder's reconstruction of frame_tp1.
            vq_loss (Tensor): The loss from the VQ layer.
            perplexity (Tensor): The perplexity of codebook usage.
            latents_e (Tensor): The output of the encoder (before quantization).
            min_encoding_indices (Tensor): The chosen codebook indices (B, H_latent*W_latent).
        """
        # Encode: frame_t and frame_tp1 -> latents_e
        # The encoder internally computes the difference and concatenates.
        latents_e = self.encoder(frame_t, frame_tp1)
        
        # Quantize: latents_e -> quantized_latents, vq_loss, perplexity, min_encoding_indices
        quantized_latents, vq_loss, perplexity, min_encoding_indices = self.quantizer(latents_e)
        
        # Decode: quantized_latents and frame_t -> reconstructed_frame_tp1
        reconstructed_frame_tp1 = self.decoder(quantized_latents, frame_t)
        
        return reconstructed_frame_tp1, vq_loss, perplexity, latents_e, min_encoding_indices

    def calculate_loss(self, 
                       frame_tp1_original: torch.Tensor, 
                       reconstructed_frame_tp1: torch.Tensor, 
                       vq_loss: torch.Tensor,
                       codebook_entropy_reg_weight: float = 0.0, # Gamma in the plan
                       min_encoding_indices: torch.Tensor | None = None, # For entropy regularization
                       num_embeddings: int | None = None # For entropy regularization
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the total loss for the VQ-VAE.

        Args:
            frame_tp1_original (Tensor): The original next frame.
            reconstructed_frame_tp1 (Tensor): The VQ-VAE's reconstruction of the next frame.
            vq_loss (Tensor): The VQ loss (commitment loss) from the quantizer.
            codebook_entropy_reg_weight (float): Weight for the codebook entropy regularization term.
            min_encoding_indices (Tensor, optional): Flat tensor of chosen codebook indices from VQ layer.
                                                   Required if codebook_entropy_reg_weight > 0.
            num_embeddings (int, optional): Total number of embeddings in the codebook.
                                             Required if codebook_entropy_reg_weight > 0.

        Returns:
            total_loss (Tensor): The combined loss.
            reconstruction_loss (Tensor): The MSE reconstruction loss.
        """
        reconstruction_loss = F.mse_loss(reconstructed_frame_tp1, frame_tp1_original)
        
        total_loss = reconstruction_loss + vq_loss

        # --- START: MODIFICATION FOR CODEBOOK ENTROPY REGULARIZATION ---
        if codebook_entropy_reg_weight > 0:
            if min_encoding_indices is None or num_embeddings is None:
                raise ValueError("min_encoding_indices and num_embeddings must be provided for codebook entropy regularization.")
            
            # Calculate empirical distribution of codebook usage
            # min_encoding_indices is (B*H_latent*W_latent,)
            counts = torch.bincount(min_encoding_indices.flatten(), minlength=num_embeddings)
            probs = counts.float() / counts.sum()
            
            # Calculate entropy: -sum(p * log(p))
            # Add epsilon to prevent log(0)
            codebook_entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            
            # We want to maximize entropy, so we minimize negative entropy.
            entropy_regularization_loss = -codebook_entropy 
            total_loss = total_loss + codebook_entropy_reg_weight * entropy_regularization_loss
        # --- END: MODIFICATION FOR CODEBOOK ENTROPY REGULARIZATION ---
            
        return total_loss, reconstruction_loss

# Next: VQVAE main model

# Further components (Encoder, Decoder, VQVAE main model) will be added below. 