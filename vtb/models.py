# vtb/models.py
import torch
import torch.nn as nn
import math
from .utils import device
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query: [B, Q_len, D] (Prompt from labels)
        # key, value: [B, K_len, D] (Features from time series)
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        output = self.norm(query + attn_output) # Residual + Norm
        return output


class ANN_MLP(nn.Module):
    """Use MLP as a surrogate brain"""
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super().__init__()
        self.init_args = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'output_dim': output_dim
        }
        self.func = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        ).to(device)
        self.to(device)
    def forward(self, x,labels=None):
        return self.func(x)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

 
    
class VTB_Transformer(nn.Module):
    def __init__(
        self,
        input_dim,     # = steps * roi_num
        steps,
        roi_num,
        d_model=768,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        use_layernorm=True,
        use_last_token = False,
        num_labels=5,  
        num_cross_layers= 2, 
        demographic_dim = 56
    ):
        super().__init__()
        self.init_args = {
            'input_dim': input_dim,
            'steps': steps,
            'roi_num': roi_num,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'use_layernorm': use_layernorm,
            'use_last_token':use_last_token,
            'num_labels': num_labels,
            'num_cross_layers': num_cross_layers,
        }
        self.steps = steps
        self.roi_num = roi_num
        self.d_model = d_model
        self.num_cross_layers = num_cross_layers
        self.use_last_token = use_last_token

        self.label_embedding = nn.Embedding(num_labels, d_model)
        self.demographic_embedding = nn.Linear(demographic_dim, d_model)

        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dropout) for _ in range(num_cross_layers)
        ])

        self.input_proj = nn.Linear(roi_num, d_model)
        self.posenc = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if use_layernorm else None
        )
                # --- Attention Pooling ---
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, roi_num)
        )
        
        # --- Parameter Initialization ---
        self.apply(self._init_weights)
        self.to(device)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0.0)
            
    def forward(self, x, labels=None, Demographic=None):
        single = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single = True
            if labels is not None and labels.dim() == 0:
                labels = labels.unsqueeze(0)
        elif x.dim() != 2:
            raise ValueError(f"Expected input dim 1 or 2, got {x.dim()}")

        B, D = x.shape

        expected = self.steps * self.roi_num
        
        if D != expected:
            
            raise ValueError(f"Input feature dim mismatch: got {D}, expected {expected}")

        x = x.view(B, self.steps, self.roi_num)   # [B, T, ROI]
        x = self.input_proj(x)                    # [B, T, d_model]
        x = self.posenc(x)                        # [B, T, d_model]


        if labels is not None or Demographic is not None:
            # Convert label IDs to embedding vectors
            # label_embeds: [B, d_model]
            label_embeds = self.label_embedding(labels)
            if Demographic is not None:
                demographic_embeds = self.demographic_embedding(Demographic)
                label_embeds = label_embeds + demographic_embeds
            # Expand dimensions to match Cross-Attention Query requirements [B, 1, d_model]
            label_embeds = label_embeds.unsqueeze(1)


            for cross_attn in self.cross_attn_layers:

                label_embeds = cross_attn(label_embeds, x, x)
            
            label_embeds = label_embeds.expand(-1, self.steps, -1)
            x = x + label_embeds

        h = self.encoder(x)                       # [B, T, d_model]

        if self.use_last_token:
            h_last = h[:,-1,:]  # [B, D]
            
        else:
            B, T, D = h.shape
            query = self.pooling_query.expand(B, -1, -1)  # [B, 1, D]
            scores = torch.bmm(query, h.transpose(1, 2)) / (D ** 0.5)  # [B, 1, T]
            attn_weights = F.softmax(scores, dim=-1)  # [B, 1, T]
            h_last = torch.bmm(attn_weights, h).squeeze(1)  # [B, D]

        y = self.head(h_last)                     # [B, ROI]

        return y[0] if single else y    



