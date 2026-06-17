import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        self.features = 1 + args.features
        self.d_model = args.d_model
        self.layers = args.e_layers
        self.n_heads = args.n_heads
        self.occurrence_branch_residual = args.occurrence_branch_residual
        self.size_branch_residual = args.size_branch_residual
        self.branch_residuals = args.branch_residuals
        self.pooling = args.pooling
        
        self.input_proj = nn.Linear(self.features, self.d_model)
        self.input_dropout = nn.Dropout(p=args.dropout)
        self.pos_encoding = nn.Parameter(torch.randn((1, self.seq_len, self.d_model)))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = self.n_heads,
            dim_feedforward = self.d_model * 4,
            dropout = args.dropout,
            batch_first = True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = self.layers
        )

        # occurrence branch
        encoder_layer_occurrence = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = self.n_heads,
            dim_feedforward = self.d_model * 4,
            dropout = args.dropout,
            batch_first = True
        )

        self.encoder_occurrence = nn.TransformerEncoder(
            encoder_layer_occurrence,
            num_layers = self.layers // 2 if self.layers >=2 else 1
        )

        self.ln_occurrence = nn.LayerNorm(self.d_model)

        self.nn_occurrence = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(self.d_model // 2, self.pred_len),
            # nn.Sigmoid(), # disabled to use BCEWithLogitsLoss instead
        )

        # demand branch
        encoder_layer_demand = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = self.n_heads,
            dim_feedforward = self.d_model * 4,
            dropout = args.dropout,
            batch_first = True
        )

        self.encoder_demand = nn.TransformerEncoder(
            encoder_layer_demand,
            num_layers = self.layers // 2 if self.layers >=2 else 1
        )

        self.ln_demand = nn.LayerNorm(self.d_model)

        self.nn_demand = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(self.d_model // 2, self.pred_len),
            nn.Softplus(),
        )

    def forward(self, batch_x, batch_y, batch_x_mark=None, batch_y_mark=None, return_hidden=False):
        if batch_x_mark is not None and self.features > 1:
            x = torch.concat((batch_x, batch_x_mark), dim=-1) # [B, L, N]
        else:
            x = batch_x

        x = self.input_proj(x) # [B, L, N] -> [B, L, D]
        x = self.input_dropout(x)
        x = x + self.pos_encoding

        x = self.encoder(x) # [B, L, D] -> [B, L, D]
    
        if self.branch_residuals or self.occurrence_branch_residual:
            out_occurrence = self.encoder_occurrence(x) + x # [B, L, D] -> [B, L, D], (using residual)
        else:
            out_occurrence = self.encoder_occurrence(x) # [B, L, D] -> [B, L, D]

        if self.branch_residuals or self.size_branch_residual:
            out_demand = self.encoder_demand(x) + x # [B, L, D] -> [B, L, D], (using residual)
        else:
            out_demand = self.encoder_demand(x) # [B, L, D] -> [B, L, D]

        out_occurrence = self.ln_occurrence(out_occurrence) # [B, L, D] -> [B, L, D]
        out_demand = self.ln_demand(out_demand) # [B, L, D] -> [B, L, D]

        if self.pooling == "mean":
            out_occurrence = out_occurrence.mean(dim=1)  # [B, L, D] -> [B, D]
            out_demand = out_demand.mean(dim=1) # [B, L, D] -> [B, D]
        elif self.pooling == "last":
            out_occurrence = out_occurrence[:, -1, :] # [B, L, D] -> [B, D]
            out_demand = out_demand[:, -1, :] # [B, L, D] -> [B, D]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
        
        if return_hidden:
            return out_occurrence, out_demand

        out_occurrence = self.nn_occurrence(out_occurrence) # [B, D] -> [B, P]
        out_demand = self.nn_demand(out_demand) # [B, D] -> [B, P]

        out_occurrence = out_occurrence.unsqueeze(-1)  # [B, P] -> [B, P, 1]
        out_demand = out_demand.unsqueeze(-1) # [B, P] -> [B, P, 1]

        return out_occurrence, out_demand
    
    def recombine(self, occurrence, demand):
        return occurrence * demand # [B, P, 1] * [B, P, 1] -> [B, P, 1]