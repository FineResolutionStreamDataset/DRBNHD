import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F


class RGCN_v0(nn.Module):

    def __init__(self, input_dim, hidden_dim, adj_matrix, recur_dropout=0, dropout=0, return_states=False, device='cpu', seed = None):
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
        self.A = torch.from_numpy(adj_matrix).float().to(device)

        self.W_graph_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_graph_h = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_graph_c = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_graph_c = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_h_cur = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.W_h_prev = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_h = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_c_cur = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.W_c_prev = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_c = nn.Parameter(torch.Tensor(hidden_dim))
        # End of new stuff

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
            c_t_cur = f_t * c_t + i_t * self.recur_dropout(g_t)
            h_t_cur = o_t * torch.tanh(c_t)

            h_graph_t = torch.tanh(self.A @ (h_t @ self.W_graph_h + self.b_graph_h))
            c_graph_t = torch.tanh(self.A @ (c_t @ self.W_graph_c + self.b_graph_c))

            # aka h_update_t / c_update_t
            h_t = torch.sigmoid(h_t_cur @ self.W_h_cur + h_graph_t @ self.W_h_prev + self.b_h)
            c_t = torch.sigmoid(c_t_cur @ self.W_c_cur + c_graph_t @ self.W_c_prev + self.b_c)

            hidden_seq.append(h_t.unsqueeze(1))
        hidden_seq = torch.cat(hidden_seq, dim=1)
        out = self.dense(hidden_seq)
        if self.return_states:
            return out, (h_t, c_t)
        else:
            return out



class RGCN_v1(nn.Module):
    # Built off of https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
    #def __init__(self, input_dim, hidden_dim, adj_matrix, recur_dropout=0, dropout=0, return_states=False, device='cuda', seed=None):
    def __init__(self, input_dim, hidden_dim, adj_matrix, recur_dropout=0, dropout=0, return_states=False, device='cpu', seed=None):


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



class RGCN_SR(nn.Module):
    def __init__(self, input_dim, hidden_dim, adj_matrix,adj_matrix_mixed=None, recur_dropout=0, dropout=0, device='cpu',
                 seed=None, save_feature=False, load_feature=False, return_state=False):

        super().__init__()
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.A = torch.from_numpy(adj_matrix).float().to(device)

        if adj_matrix_mixed is not None:
            self.adj_matrix_mixed = torch.from_numpy(adj_matrix_mixed).float().to(device) #2023.10

        self.save_feature = save_feature #2023.10
        self.load_feature = load_feature #2023.10
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
        self.dense_fusion = nn.Linear(hidden_dim*2, 1)
        self.device = device
        self.return_state = return_state
#        if adj_matrix_mixed is not None:
#            self.adj_matrix_mixed = nn.Parameter(torch.from_numpy(adj_matrix_mixed).float().to(device))
#        if adj_matrix_mixed is not None:
#            shape = adj_matrix_mixed.shape
#            self.adj_matrix_mixed = nn.Parameter(torch.rand(shape).float().to(device))
#        if adj_matrix_mixed is not None:  #DRB+NHD
#            self.adj_matrix_mixed = nn.Parameter(torch.rand(63,455).float().to(device))


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

        if self.save_feature:
            B, T, F = hidden_seq.shape
            hidden_seq_reshaped = hidden_seq.view(B, -1)
            result_reshaped = torch.matmul(self.adj_matrix_mixed, hidden_seq_reshaped)
            hidden_seq = result_reshaped.view(self.adj_matrix_mixed.shape[0],T,F)



        if self.load_feature:
            hidden_seq_NHM = torch.load(self.load_feature)
            hidden_seq = torch.cat((hidden_seq, hidden_seq_NHM), dim=-1)
            out = self.dense_fusion(hidden_seq)
        else:
            out = self.dense(hidden_seq)  # torch.Size([13, 365, 1]) old #torch.Size([63, 365, 1]) new


        if self.return_state:
            return out, hidden_seq
        else:
            return out



class Task0Module(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)
    def forward(self, hidden_seq):
        out = self.dense(hidden_seq)
        return out

'''
class Task1Module(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, hidden_seq):
        hidden_seq_transposed = hidden_seq.transpose(0, 1)
        attn_output, _ = self.attention(hidden_seq_transposed, hidden_seq_transposed, hidden_seq_transposed)
        attn_output = attn_output.transpose(0, 1)
        attn_output = self.norm(attn_output.transpose(1, 2))
        attn_output = attn_output.transpose(1, 2)
        out = self.dense(attn_output)
        return out
'''

'''
class CrossBatchInteraction(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 用于跨批次交互的多头注意力机制
        self.interaction = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)

    def forward(self, hidden_seq):
        # hidden_seq 形状为 [batch_size, seq_length, hidden_size]
        batch_size, seq_length, hidden_size = hidden_seq.size()
        # 直接使用 batch_size，不需要 total_batches
        # 重塑 hidden_seq 以适应多头注意力
        hidden_seq_reshaped = hidden_seq.transpose(0, 1)  # [seq_length, batch_size, hidden_size]
        # 应用多头注意力机制
        interaction_output, _ = self.interaction(hidden_seq_reshaped, hidden_seq_reshaped, hidden_seq_reshaped)
        # 输出已经是 [seq_length, batch_size, hidden_size]，无需再次重塑
        interaction_output = interaction_output.transpose(0, 1)  # 转换回 [batch_size, seq_length, hidden_size]
        return interaction_output

class Task1Module(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 初始化跨批次交互层时，不再传入 total_batches
        self.cross_batch_interaction = CrossBatchInteraction(hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        self.global_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        # 特征维度翻倍处理
        self.norm = nn.BatchNorm1d(hidden_size * 2)
        self.dense = nn.Linear(hidden_size * 2, 1)

    def forward(self, hidden_seq):
        # 局部（批次内）信息处理
        hidden_seq_transposed = hidden_seq.transpose(0, 1)
        local_attn_output, _ = self.attention(hidden_seq_transposed, hidden_seq_transposed, hidden_seq_transposed)
        # 跨批次交互
        cross_batch_out = self.cross_batch_interaction(hidden_seq)
        # 融合局部注意力输出和跨批次交互输出
        combined_attn_output = torch.cat([local_attn_output.transpose(0, 1), cross_batch_out], dim=-1)
        # 应用后续层
        combined_attn_output = self.norm(combined_attn_output.transpose(1, 2)).transpose(1, 2)
        out = self.dense(combined_attn_output)
        return out
'''

'''
class CrossBatchInteraction(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.interaction = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        # 添加Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_seq):
        batch_size, seq_length, hidden_size = hidden_seq.size()
        hidden_seq_reshaped = hidden_seq.transpose(0, 1)
        hidden_seq_reshaped = self.layer_norm(hidden_seq_reshaped)  # 应用LayerNorm
        interaction_output, _ = self.interaction(hidden_seq_reshaped, hidden_seq_reshaped, hidden_seq_reshaped)
        interaction_output = interaction_output.transpose(0, 1)  # 转换回 [batch_size, seq_length, hidden_size]
        # 应用残差连接和再次归一化
        interaction_output = hidden_seq + interaction_output  # 添加残差连接
        interaction_output = self.layer_norm(interaction_output)  # 再次应用LayerNorm
        return interaction_output

class Task1Module(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.cross_batch_interaction = CrossBatchInteraction(hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        # 在注意力层后添加LayerNorm
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.global_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        # 在全局注意力层后添加LayerNorm
        self.global_attention_layer_norm = nn.LayerNorm(hidden_size)
        # 调整后的特征维度，去除特征维度翻倍
        self.norm = nn.LayerNorm(hidden_size)  # 使用LayerNorm替代BatchNorm1d
        self.dense = nn.Linear(hidden_size, 1)
        # 添加Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_seq):
        hidden_seq_transposed = hidden_seq.transpose(0, 1)
        local_attn_output, _ = self.attention(hidden_seq_transposed, hidden_seq_transposed, hidden_seq_transposed)
        local_attn_output = local_attn_output.transpose(0, 1)  # 转换回原始维度
        local_attn_output = self.attention_layer_norm(hidden_seq + local_attn_output)  # 添加残差连接和LayerNorm
        cross_batch_out = self.cross_batch_interaction(hidden_seq)
        # 直接合并而不是扩大特征维度
        combined_attn_output = local_attn_output + cross_batch_out  # 融合局部注意力和跨批次交互输出
        combined_attn_output = self.norm(combined_attn_output)  # 应用LayerNorm
        combined_attn_output = self.dropout(combined_attn_output)  # 应用Dropout
        out = self.dense(combined_attn_output)
        return out
'''


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
        # 添加残差连接和BatchNorm
        local_attn_output = self.attention_batch_norm((hidden_seq + local_attn_output).transpose(1, 2)).transpose(1, 2)
        cross_batch_out = self.cross_batch_interaction(hidden_seq)
        # 融合局部注意力和跨批次交互输出
        combined_attn_output = local_attn_output + cross_batch_out
        # 应用BatchNorm
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
    def __init__(self, input_dim, hidden_dim, adj_matrix, adj_matrix_NHD, adj_matrix_mixed=None, recur_dropout=0,
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

