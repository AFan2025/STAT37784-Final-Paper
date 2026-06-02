import torch
import torch.nn as nn
import numpy as np
import math

# modulation for the gating mechanism in Adaptive Layer Norm
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class BrainConvolutionalEncoder(nn.Module):
    """
    Docstring for BrainConvolutionalEncoder
    Convolutional Encoder for processing brain embeddings, takes the BxSx2x16x8 input and encodes it into a B x sequence_encoded_dim x z_brain_dim representation.
    The dimensions for each datapoint are 2 channels (tx1, spikePow) with 16x8 spatial dimensions representing the electrode grid for 6v region.
    For training speed for cross-attention conditioning in the Phoneme DiT model, default behaviour sets sequence_encoded_dim == phoneme_encoded_dim.
    However, these can be set differently for ablation studies, such as setting sequence_encoded_dim == S to see if higher capacity brain embeddings help.
    """
    def __init__(self, 
                 input_channels=2, 
                 sequence_encoded_dim=128, 
                 z_brain_dim=256, use_layer_norm = True,
                 mlp_num_hidden_layers = 2):
        super(BrainConvolutionalEncoder, self).__init__()
        self.input_channels = input_channels
        self.sequence_encoded_dim = sequence_encoded_dim
        self.z_brain_dim = z_brain_dim
        self.use_layer_norm = use_layer_norm

        # Convolutional layers to process spatial dimensions
        self.conv_layers = nn.Sequential( #(b, s, 2, 16, 8)
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=2, padding=1), # (b, s, 16, 8, 4)
            # nn.BatchNorm2d(16),
            nn.GroupNorm(num_groups=8, num_channels=16), # using group norm instead of batch norm for better performance on smaller batch sizes
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2), # (b, s, 16, 8, 4)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # (b, s, 32, 4, 2)
            nn.GroupNorm(num_groups=8, num_channels=32), # using group norm instead of batch norm for better performance on smaller batch sizes
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            # nn.MaxPool2d(kernel_size=2), # (b, s, 32, 4, 2)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # (b, s, 64, 4, 2)
            nn.GroupNorm(num_groups=8, num_channels=64), # using group norm instead of batch norm for better performance on smaller batch sizes
            # supposedly ^^ this 64 conv2d would just get averaged away anyways, so possibly changing to reduce parameters might help.

            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)) # (b, s, 64, 1, 1)
        )

        # Convolutional Layer converstions
        # (b, s, 2, 16, 8)
        # (b, s, 16, 8, 4)
        # (b, s, 32, 4, 2)
        # (b, s, 64, 4, 2)
        # (b, s, 64, 1, 1)

        if mlp_num_hidden_layers > 1:
        # Linear layer to project to desired z_brain_dim
            self.ffn = nn.Sequential(
                nn.Flatten(start_dim=2), # (b, s, 64)
                nn.Linear(64, 128), # (b, s, 128)
                nn.ReLU(inplace=True),
                nn.Dropout(p = 0.2),
                nn.Linear(128, z_brain_dim) # (b, s, z_brain_dim)
                )
        else:
            self.ffn = nn.Sequential(
                nn.Flatten(start_dim = 2),
                nn.Linear(64, z_brain_dim) # (b, s, z_brain_dim)
            )
        
        #maybe a layer here for better convergence?
        self.ln_self = nn.LayerNorm(z_brain_dim)
        
    def forward(self, x):
        # x.shape = (b, s, 2, 16, 8)
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w) # (b*s, 2, 16, 8)
        x = self.conv_layers(x)
        x = x.view(b, s, -1) # (b, s, 64)
        x = self.ffn(x)

        # decide here if using layer norm or not
        # if self.use_layer_norm:
        x = self.ln_self(x)

        return x

#  https://github.com/facebookresearch/DiT/blob/main/models.py#L292
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256): 
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class SinPosEmbedding(nn.Module): # use until ROPE is implemented, for simplicity for now
    def __init__(self, max_len, d_model):
        super(SinPosEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x): #This is learned embeddings not sinusoidal 
        # x.shape = (b, s, d_model)
        b, s, d_model = x.shape
        positions = torch.arange(0, s, device=x.device).unsqueeze(0).expand(b, s) # (b, s)
        pos_emb = self.pos_embedding(positions) # (b, s, d_model)
        return pos_emb

class PhonemeDiTBlock(nn.Module):
    """
    Docstring for PhonemeDiTBlock
    Single block of the Phoneme Diffusion-Transformer model
    This block works as a forward pass for a single transformer block to operate Phoneme to Phoneme diffusion.
    The block flows through a self-attention layer, optional cross-attention layer (for conditioning), and a feed-forward network (MLP).
    The cross-attention layer will be used to condition in the brain embeddings during fine-tuning for transfer learning the brain-to-text task.
    Args:
        hidden_dim (int): Dimension of the hidden representations.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of the MLP hidden dimension to the hidden_dim (default set to 4 according to online as best practice)
        cross_attention (bool): Whether to include cross-attention for conditioning.
    """
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, use_cross_attention=False, cond_dim=None):
        super(PhonemeDiTBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        if cond_dim is not None:
            kdim = cond_dim
            vdim = cond_dim
            self.cond_dim = cond_dim
        else:
            kdim = hidden_dim
            vdim = hidden_dim
            self.cond_dim = hidden_dim

        # whether or not to use cross-attention for conditioning
        self.use_cross_attention = use_cross_attention

         # LayerNorm layers
        self.ln_self = nn.LayerNorm(hidden_dim)
        self.ln_ff = nn.LayerNorm(hidden_dim)

        # Multi-head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Cross-Attention (also can ablate later using Adaptive Layer Norm for conditioning instead of cross-attention)
        if use_cross_attention:
            self.ln_cross = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, kdim=kdim, vdim=vdim, num_heads=num_heads, batch_first=True)
            
            # Gate for the cross attention adaptive norm
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            #for speed and efficiency, can ablate later for higher accuracy
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )

        # adaptive layer norm (claude says its "free")
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

        # alternative option outside of zero weighting (9 gate)
        # This is fiddly because when fine tuning with additional, you need to keep the original learned weights for the 0-2, 6-8 indices and bring them over
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim, 9 * hidden_dim, bias=True)
        # )

    def forward(self, x, c, x_mask=None, z_brain=None, cond_mask=None):
        #inverting masks for padding
        # if x_mask is not None:
        #     x_mask = ~x_mask
        # if cond_mask is not None:
        #     cond_mask = ~cond_mask

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)

        # Self-Attention block with adaptive layer norm modulation
        x_norm = self.ln_self(x)
        x_mod = modulate(x_norm, shift_msa, scale_msa)
        attn_output, _ = self.attn(x_mod, x_mod, x_mod, key_padding_mask=x_mask)
        attn_output = gate_msa.unsqueeze(1) * attn_output
        x = x + attn_output

        # Cross-Attention block (if conditioning is provided)
        if self.use_cross_attention and z_brain is not None:
            x_norm = self.ln_cross(x)
            cross_attn_output, _ = self.cross_attn(x_norm, z_brain, z_brain, key_padding_mask=cond_mask)
            x = x + self.cross_attn_gate*cross_attn_output

        # if self.use_cross_attention and z_brain is not None:
        #     x_norm = self.ln_cross(x)
        #     x_mod = modulate(x_norm, shift_cross, scale_cross)
        #     cross_attn_output, _ = self.cross_attn(x_mod, z_brain, z_brain, key_padding_mask=cond_mask)
        #     cross_attn_output = gate_cross.unsqueeze(1) * cross_attn_output
        #     x = x + cross_attn_output

        # Feed-Forward block with adaptive layer norm modulation
        x_norm = self.ln_ff(x)
        x_mod = modulate(x_norm, shift_mlp, scale_mlp)
        ff_output = gate_mlp.unsqueeze(1) * self.mlp(x_mod)
        x = x + ff_output
        return x
        
class FinalLayer(nn.Module): # completely optional, maybe not even added in
    """
    Final layer that projects the denoised latent vectors back into the tokens in accordance to .env vocab size
    Borrowed from #  https://github.com/facebookresearch/DiT/blob/main/models.py#L125
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class PhonemeDiT(nn.Module):
    """
    Full architecture of everything put together
    """

    def __init__(self, 
                d_model = 1024, #dimension of the model
                vocab_size = 24, #vocab size
                depth = 6, # number of blocks
                max_len = 128, #maximum number of phonemes per data
                num_heads = 8, #number of attention heads per block 
                mlp_ratio=4.0, #ratio of how large the up proj of the block MLPs compared to d_model
                use_cross_attention=False, #whether or not the model is conditioned vs unconditional (unconditional pretraining vs brain conditioned fine tuning)
                frequency_embedding_size=256, # frequency embedding size for timestep embedding, idk what it means but taken from original DiT
                brain_enc_input_channels=2, #brain data input dimensions, should remain 2 for both channels of data spikePow and other
                sequence_encoded_dim=128, #unused rn but use if we want to add temporal downsampling
                z_brain_dim=1024, #final output dimension of z_brain
                brain_enc_use_layer_norm = True, #whether the brain encoder uses layer norm
                brain_enc_mlp_num_hidden_layers = 2, #whether the brain uses MLP (only accepts 1 or 2, will fix later TODO)
                use_final_layer = False, # whether or not to use the specialized final layer
                decoder_approach = "nn"
                ):
        super().__init__()
        # Param Inits
        self.d_model = d_model
        self.use_final_layer = use_final_layer
        self.use_cross_attention = use_cross_attention

        # Embedding Layer
        self.x_embedder = nn.Embedding(vocab_size,d_model)

        # Layer inits
        if self.use_cross_attention:
            self.brain_encoder = BrainConvolutionalEncoder(brain_enc_input_channels,
                                                        sequence_encoded_dim,
                                                        z_brain_dim,
                                                        brain_enc_use_layer_norm,
                                                        brain_enc_mlp_num_hidden_layers)
        self.t_embedder = TimestepEmbedder(d_model, frequency_embedding_size)
        self.pos_embedder = SinPosEmbedding(max_len=max_len, d_model=d_model)
        if self.use_final_layer:
            self.final_layer = FinalLayer(d_model)

        self.blocks = nn.ModuleList([
            PhonemeDiTBlock(d_model,
                            num_heads, 
                            mlp_ratio=mlp_ratio, 
                            use_cross_attention = use_cross_attention, 
                            cond_dim = z_brain_dim) for _ in range(depth)
        ])

        self.decoder = DecoderLayer(d_model, vocab_size, self.x_embedder, decoder_approach)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize all model weights with DiT-style defaults:
        - Kaiming init for conv layers (ReLU stacks)
        - Xavier init for linear/attention projections
        - Unit init for norm layers
        - N(0, 0.02) for embedding tables
        - Zero-init adaptive modulation heads (adaLN-Zero behavior)
        Generated by GPT-5.3-Codex
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)
                # Cross-attention with separate k/v dims exposes separate projection weights.
                if hasattr(module, 'q_proj_weight') and module.q_proj_weight is not None:
                    nn.init.xavier_uniform_(module.q_proj_weight)
                if hasattr(module, 'k_proj_weight') and module.k_proj_weight is not None:
                    nn.init.xavier_uniform_(module.k_proj_weight)
                if hasattr(module, 'v_proj_weight') and module.v_proj_weight is not None:
                    nn.init.xavier_uniform_(module.v_proj_weight)

        # DiT-style timestep MLP init.
        for layer in self.t_embedder.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

        # adaLN-Zero: start modulation heads from zero so residual blocks behave near-identity initially.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        if hasattr(self, 'final_layer'):
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
            nn.init.zeros_(self.final_layer.linear.weight)
            if self.final_layer.linear.bias is not None:
                nn.init.zeros_(self.final_layer.linear.bias)

    def embed_tok(self, x):
        return self.x_embedder(x) * math.sqrt(self.d_model)
    
    @torch.no_grad()
    def decode_tok(self, x):
        return self.decoder(x)
    
    def forward(self, x, x_mask, t, brain_data, brain_mask): #TODO finish the forward pass
        """
        Forward pass of full architecture PhonemeDiT
        """
        # Invert masks once for nn.MultiheadAttention convention
        # Our convention: True = real token, False = pad
        # MHA convention: True = ignore, False = attend
        attn_mask = ~x_mask if x_mask is not None else None
        attn_cond_mask = ~brain_mask if brain_mask is not None else None

        if self.use_cross_attention:
            brain_enc = self.brain_encoder(brain_data)
            brain_global = brain_enc.mean(dim=1)
        
        x = x + self.pos_embedder(x)  
        t = self.t_embedder(t)                   
        c = t
        if self.use_cross_attention:
            c += brain_global                          
        for block in self.blocks:
            if self.use_cross_attention:
                x = block(x, c, attn_mask, brain_enc, attn_cond_mask)
            else:
                x = block(x, c, attn_mask)
        if self.use_final_layer:
            x = self.final_layer(x, c)
        return x

class DecoderLayer(nn.Module):
    """
    Decodes representations back into phonemes, can use either learned or nearest neighbor decoding, dependent on amount of time to train. 
    """
    def __init__(self, d_model, vocab_size, embedding_layer, approach="nn"):
        super().__init__()  # also missing this
        self.approach = approach
        self.embedding_layer = embedding_layer
        self.unembedding_layer = nn.Embedding(d_model, vocab_size)
        self.softmax_layer = nn.Softmax(dim = 1)

    def nn_decoding(self, x_clean):
        # Nearest-neighbor in embedding table
        distances = torch.cdist(x_clean, self.embedding_layer.weight)
        return distances.argmin(dim=-1)

    def learned_decoding(self, x):
        logits = self.unembedding_layer(x)
        return self.softmax_layer(logits)

    def forward(self, x):
        if self.approach == "nn":
            return self.nn_decoding(x)
        elif self.approach == "learned":
            return self.learned_decoding(x)
        else:
            raise ValueError("Please choose decoding approach between learned or nearest neighbor(nn)")
