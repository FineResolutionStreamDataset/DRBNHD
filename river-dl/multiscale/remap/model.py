import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter







class RGCN_v1(nn.Module):
    # Built off of https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
    # def __init__(self, input_dim, hidden_dim, adj_matrix, recur_dropout=0, dropout=0, return_states=False, device='cuda', seed=None):
    def __init__(self, input_dim, hidden_dim, adj_matrix, remap_matrix, recur_dropout=0, dropout=0, return_states=False, device='cpu',
                 seed=None):

        """
        @param input_dim: [int] number input feature
        @param hidden_dim: [int] hidden size
        @param adj_matrix: Distance matrix for graph convolution
        @param recur_dropout: [float] fraction of the units to drop from the cell update vector. See: https://arxiv.org/abs/1603.05118
        @param dropout: [float] fraction of the units to drop from the input
        @param return_states: [bool] If true, returns h and c states as well as predictions
        """
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        super().__init__()

        # New stuff
        self.A = torch.from_numpy(adj_matrix).float().to(device)  # provided at initialization
        # parameters for mapping graph/spatial data
        self.weight_q = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_q = nn.Parameter(torch.Tensor(hidden_dim))

        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.weight_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()

        self.dropout = nn.Dropout(dropout)
        self.recur_dropout = nn.Dropout(recur_dropout)

        self.dense = nn.Linear(hidden_dim, 1)
        self.return_states = return_states
        if remap_matrix is not None:
            self.remap_matrix = torch.from_numpy(remap_matrix).float().to(device)

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        x = self.dropout(x)
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            q_t = torch.tanh(h_t @ self.weight_q + self.bias_q)
            c_t = f_t * (c_t + self.A @ q_t) + i_t * self.recur_dropout(g_t)  # note: self.A @ q_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))
        hidden_seq = torch.cat(hidden_seq, dim=1)
        out = self.dense(hidden_seq)
        if self.return_states:
            return out, (h_t, c_t)
        else:
            return out











class Task0Module(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)
    def forward(self, hidden_seq):
        out = self.dense(hidden_seq)
        return out




class CrossBatchInteraction(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.interaction = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        # BatchNorm
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, hidden_seq):
        hidden_seq_reshaped = hidden_seq.transpose(0, 1)
        interaction_output, _ = self.interaction(hidden_seq_reshaped, hidden_seq_reshaped, hidden_seq_reshaped)
        interaction_output = interaction_output.transpose(0, 1)
        # Resnet+BatchNorm
        interaction_output = self.batch_norm((hidden_seq + interaction_output).transpose(1, 2)).transpose(1, 2)
        return interaction_output

class Task1Module(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.cross_batch_interaction = CrossBatchInteraction(hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)

        self.attention_batch_norm = nn.BatchNorm1d(hidden_size)
        self.global_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)

        self.global_attention_batch_norm = nn.BatchNorm1d(hidden_size) #BatchNorm

        self.norm = nn.BatchNorm1d(hidden_size)  # 使用BatchNorm1d
        self.dense = nn.Linear(hidden_size, 1)
        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_seq):
        hidden_seq_transposed = hidden_seq.transpose(0, 1)
        local_attn_output, _ = self.attention(hidden_seq_transposed, hidden_seq_transposed, hidden_seq_transposed)
        local_attn_output = local_attn_output.transpose(0, 1)
        local_attn_output = self.attention_batch_norm((hidden_seq + local_attn_output).transpose(1, 2)).transpose(1, 2)
        cross_batch_out = self.cross_batch_interaction(hidden_seq)
        combined_attn_output = local_attn_output + cross_batch_out
        combined_attn_output = self.norm(combined_attn_output.transpose(1, 2)).transpose(1, 2)
        combined_attn_output = self.dropout(combined_attn_output)
        out = self.dense(combined_attn_output)
        return out





class Task2Module(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(hidden_size * 2)
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()  # Using GELU for better performance
        self.dropout = nn.Dropout(0.1)  # Regularization
        self.dense2 = nn.Linear(hidden_size, 1)
        self.residual = nn.Linear(input_size, hidden_size)  # For residual connection

    def forward(self, combined_hidden_seq):
        normalized_seq = self.norm1(combined_hidden_seq.transpose(1, 2)).transpose(1, 2)
        x = self.activation(self.dense1(normalized_seq))
        x = self.dropout(x)
        x = x + self.residual(combined_hidden_seq)
        out = self.dense2(x)
        return out


class SharedModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0, recur_dropout=0, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.weight_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.weight_q = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_q = nn.Parameter(torch.Tensor(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.recur_dropout = nn.Dropout(recur_dropout)
        self.device = device
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.weight_q)
        nn.init.zeros_(self.bias_q)

    def forward(self, x, adj_matrix, init_states=None):
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                    torch.zeros(bs, self.hidden_size).to(x.device)) if init_states is None else init_states
        x = self.dropout(x)
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),
                torch.sigmoid(gates[:, HS:HS * 2]),
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),
            )
            q_t = torch.tanh(h_t @ self.weight_q + self.bias_q)
            c_t = f_t * (c_t + adj_matrix @ q_t) + i_t * self.recur_dropout(g_t)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))
        return torch.cat(hidden_seq, dim=1)


class RGCN_Multi(nn.Module):
    def __init__(self, input_dim, hidden_dim, adj_matrix, adj_matrix_NHD, adj_matrix_mixed=None, remap_matrix=None, recur_dropout=0,
                 dropout=0, device='cpu', seed=None):
        super().__init__()
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Adjacency matrices
        self.A = torch.from_numpy(adj_matrix).float().to(device)
        self.A_NHD = torch.from_numpy(adj_matrix_NHD).float().to(device)
        if adj_matrix_mixed is not None:
            self.adj_matrix_mixed = torch.from_numpy(adj_matrix_mixed).float().to(device)
        if remap_matrix is not None:
            self.remap_matrix = torch.from_numpy(remap_matrix).float().to(device)
        # Shared layer parameters
        self.shared_layer = SharedModule(input_dim, hidden_dim, dropout, recur_dropout, device)

        # Task-specific layers
        self.task0_module = Task0Module(hidden_dim)
        self.task1_module = Task1Module(hidden_dim)
        self.task2_module = Task2Module(hidden_dim * 2, hidden_dim)

        # Device configuration
        self.device = device

    def get_shared_parameters(self):
        return [param for name, param in self.named_parameters() if 'shared_layer' in name]

    def get_task0_parameters(self):
        return [param for name, param in self.named_parameters() if 'task0_module' in name]

    def get_task1_parameters(self):
        return [param for name, param in self.named_parameters() if 'task1_module' in name]

    def get_task2_parameters(self):
        return [param for name, param in self.named_parameters() if 'task2_module' in name]

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


    # Default forward function (could be considered for task 0 or general purpose)
    def forward_task0(self, x, init_states=None):
        hidden_seq = self.shared_layer(x, self.A, init_states)
        out = self.task0_module(hidden_seq)
        return out

    # Forward function for task 1
    def forward_task1(self, x, init_states=None):
        hidden_seq = self.shared_layer(x, self.A, init_states)
        if hasattr(self, 'adj_matrix_mixed'):
            B, T, F = hidden_seq.shape
            hidden_seq_reshaped = hidden_seq.view(B, -1)
            result_reshaped = torch.matmul(self.adj_matrix_mixed, hidden_seq_reshaped)
            hidden_seq = result_reshaped.view(self.adj_matrix_mixed.shape[0], T, F)
        out = self.task1_module(hidden_seq)
        return out, hidden_seq

    # Forward function for task 2
    def forward_task2(self, x, task1_hidden_seq, init_states=None):
        hidden_seq = self.shared_layer(x, self.A_NHD, init_states)
        combined_hidden_seq = torch.cat((task1_hidden_seq, hidden_seq), dim=-1)
        out = self.task2_module(combined_hidden_seq)
        return out