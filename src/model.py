import torch
from src.mdeling_t5 import T5ForConditionalGeneration

import torch.nn.functional as F
from torch import nn

import copy
import math



class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, 
                question_ids=None, question_attention_mask=None,
                psg_ids=None, psg_attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
            
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            question_ids=question_ids, question_attention_mask=question_attention_mask,
            psg_ids=psg_ids, psg_attention_mask=psg_attention_mask,            
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask,
                 max_length,
                 question_ids=None, question_attention_mask=None,
                 psg_ids=None, psg_attention_mask=None,                 ):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1), 
            psg_ids=psg_ids, psg_attention_mask=psg_attention_mask,          #
            max_length=max_length,
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self.wrap_encoder()



class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        self.main_input_name = encoder.main_input_name
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

        setattr(self.encoder.config,'attention_probs_dropout_prob', 0.1)
        setattr(self.encoder.config,'hidden_dropout_prob', 0.1)      

        import copy
        self.inter_attn = True
        if self.inter_attn:
            row_psg_attn_config = copy.deepcopy(self.encoder.config)
            row_psg_attn_config.num_attention_heads = 1
            self.row_psg_attn = SelfAttention(row_psg_attn_config)
            
      
            
            self.seq_len = 10#1000        300 + 700
            self.row_psg_attn_position = PositionalEncoding(
                self.seq_len, self.encoder.config.hidden_size
            )       
            self.row_psg_attn_gate = nn.Parameter(torch.tensor([0.]))
            
            ####
            for n, p in self.row_psg_attn.named_parameters():
                # print(f"Initialize '{n}'")
                if "bias" not in n:
                    p.data.normal_(mean=0.0, std=0.02)
                else:
                    p.data.zero_()        
            ####            
            

            self.dim_lower = nn.Linear(768,128)
            self.dim_higher = nn.Linear(128,768)
            
            self.mega_layer = MultiHeadedEMA(
                dim = 128,
                heads = 1,
                bidirectional = not True,
                dim_head = 1
            )            
            
            ####
            for n, p in self.mega_layer.named_parameters():
                # print(f"Initialize '{n}'")
                if "bias" not in n:
                    p.data.normal_(mean=0.0, std=0.02)
                else:
                    p.data.zero_()        
            ####            
            ####
            for n, p in self.dim_lower.named_parameters():
                # print(f"Initialize '{n}'")
                if "bias" not in n:
                    p.data.normal_(mean=0.0, std=0.02)
                else:
                    p.data.zero_()        
            ####    
            ####
            for n, p in self.dim_higher.named_parameters():
                # print(f"Initialize '{n}'")
                if "bias" not in n:
                    p.data.normal_(mean=0.0, std=0.02)
                else:
                    p.data.zero_()        
            ####                  
            # self.mega_layer =  SelfAttention(row_psg_attn_config)
            
    def forward(self, input_ids=None, attention_mask=None,
                question_ids=None, question_attention_mask=None,
                psg_ids=None, psg_attention_mask=None,                 **kwargs,):
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages        
        
        
        row_num = self.n_passages
        row_len = total_length // row_num
                
        psg_num = psg_ids.shape[1]
        psg_len = psg_ids.shape[2]
        

        input_ids = input_ids.view(bsz*row_num, passage_length)
        attention_mask = attention_mask.view(bsz*row_num, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        
        row_hidden = outputs.last_hidden_state
        psg_hidden = self.encoder(input_ids=psg_ids.view(bsz*row_num, psg_len), 
                                       attention_mask=psg_attention_mask.view(bsz*row_num, psg_len), **kwargs).last_hidden_state        

        row_hidden_temp = row_hidden.view(bsz*row_num, row_len, 768)
        psg_hidden_temp = psg_hidden.view(bsz*row_num, psg_len, 768)
        attention_mask_temp = attention_mask.view(bsz*row_num, row_len)
        psg_attention_mask_temp = psg_attention_mask.view(bsz*row_num, psg_len)
        

        cat_embeds =  torch.cat((row_hidden_temp, psg_hidden_temp), dim=1)
        cat_masks = torch.cat((attention_mask_temp, psg_attention_mask_temp), dim=1)

        # row-psg inter attention
        row_psg_attn_embeds = self.row_psg_attn(self.row_psg_attn_position(cat_embeds), mask=cat_masks.unsqueeze(1)) 
               
        decoder_input_level1 = row_psg_attn_embeds * self.row_psg_attn_gate.tanh() + cat_embeds
        
        row_psg_attn_embeds = row_psg_attn_embeds  # residual connection
        

        ##################
        row_psg_attn_embeds_temp = row_psg_attn_embeds.view(bsz, row_num,  row_len+psg_len, 768)
        row_psg_attn_embeds_temp = row_psg_attn_embeds_temp.view(bsz, -1, 768)
        
       
        
        # low dim EMa
        aggre_attn_output = self.dim_lower(row_psg_attn_embeds_temp)
        aggre_attn_output = self.mega_layer(aggre_attn_output)
        aggre_attn_output = self.dim_higher(aggre_attn_output)

        
        outputs.last_hidden_state = decoder_input_level1.view(bsz, -1, 768)
        


        attention_mask = cat_masks.view(bsz,-1)

        setattr(outputs,'encoder_attention_mask', attention_mask)  

        
        setattr(outputs,'attention_mask_level_1', None)  
        setattr(outputs,'attention_mask_level_2', None)  
        setattr(outputs,'level_1_hidden_state', aggre_attn_output.view(bsz, -1, 768))#) 
        setattr(outputs,'level_2_hidden_state', None) 
        
        return outputs

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block






def attention(
    query, key, value, mask=None, dropout=None, bias=None, attn_bias_type=None
):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if bias is not None:
        if attn_bias_type == "dot":
            assert scores.size(0) == bias.size(0) and scores.size(-1) == bias.size(-1)
            scores = scores + bias[:, None, None, :]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -10000.0)

    p_attn = torch.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, config, attn_bias_type=None):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.d_k = config.hidden_size // config.num_attention_heads
        self.h = config.num_attention_heads
        self.linears = self.clones(nn.Linear(config.hidden_size, config.hidden_size), 4)
        self.attn = None
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attn_bias_type = attn_bias_type

    def clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, query, key, value, mask=None, bias=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        def gate(x, p):
            assert x.size(0) == p.size(0) and x.size(1) == p.size(-1)
            return x + self.dropout(p.unsqueeze(-1) * x)

        if bias is not None:
            if self.attn_bias_type == "key_only":
                key = gate(key, bias)
            elif self.attn_bias_type == "value_only":
                value = gate(value, bias)
            elif self.attn_bias_type == "both":
                key = gate(key, bias)
                value = gate(value, bias)

        n_b = query.size(0)
        query, key, value = [
            lin(x).view(n_b, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query,
            key,
            value,
            mask=mask,
            dropout=self.dropout,
            bias=bias,
            attn_bias_type=self.attn_bias_type,
        )

        x = x.transpose(1, 2).contiguous().view(n_b, -1, self.h * self.d_k)
        return self.linears[-1](x)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadedAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, mask=None, bias=None):
        return x + self.dropout(self.self_attn(x, x, x, mask, bias))
    
    
import torch.nn as nn
import torch
import math


    
class PositionalEncoding(nn.Module):
    def __init__(self, num_positions, d_model):
        super().__init__()
        pe = torch.zeros(num_positions, d_model)
        position = torch.arange(0, num_positions).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe      
    
    
    
    
    
import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.fft import rfft, irfft

from einops import rearrange
from scipy.fftpack import next_fast_len

# functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def default(val, d):
    return val if exists(val) else d

def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    # O(N log(N)) 1d convolution using some fourier trick

    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = rfft(x, n = fast_len, dim = dim)
    f_weight = rfft(weights, n = fast_len, dim = weight_dim)

    f_v_weight = f_x * append_dims(f_weight.conj(), weight_dim - dim)
    out = irfft(f_v_weight, fast_len, dim = dim)
    out = out.roll(-1, dims = (dim,))

    indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    out = out.index_select(dim, indices)
    return out

# positional bias for single-headed attention

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# classes

class LaplacianAttnFn(nn.Module):
    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt(0.25 * math.pi)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

class SingleHeadedAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_qk,
        dim_value,
        causal = False,
        laplacian_attn_fn = False
    ):
        super().__init__()
        self.causal = causal
        self.laplacian_attn_fn = laplacian_attn_fn

        self.attn_fn = partial(F.softmax, dim = -1) if not laplacian_attn_fn else LaplacianAttnFn()

        self.rel_pos_bias = T5RelativePositionBias(causal = causal, scale = dim_qk ** 0.5)

        self.to_qk = nn.Sequential(
            nn.Linear(dim, dim_qk),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(dim_qk, heads = 2)

        self.to_v = nn.Sequential(
            nn.Linear(dim, dim_value),
            nn.SiLU()
        )

    def forward(self, x, v_input = None):
        seq_len, dim, device, dtype = *x.shape[-2:], x.device, x.dtype

        v_input = default(v_input, x)

        qk, v = self.to_qk(x), self.to_v(v_input)
        q, k = self.offsetscale(qk)

        scale = (seq_len ** -1) if self.laplacian_attn_fn else (dim ** -0.5)

        sim = einsum('b i d, b j d -> b i j', q, k) * scale

        sim = sim + self.rel_pos_bias(sim)

        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).triu(1)

        if self.causal and not self.laplacian_attn_fn:
            # is softmax attention and using large negative value pre-softmax
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = self.attn_fn(sim)

        if self.causal and self.laplacian_attn_fn:
            # if using laplacian attention function, zero out upper triangular with 0s
            attn = attn.masked_fill(causal_mask, 0.)

        return einsum('b i j, b j d -> b i d', attn, v)

class MultiHeadedEMA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        bidirectional = False,
        dim_head = None
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.expansion = nn.Parameter(torch.randn(heads * (2 if bidirectional else 1), dim))
        self.reduction = nn.Parameter(torch.randn(heads * (2 if bidirectional else 1), dim))

        # learned alpha and dampening factors

        self.alphas = nn.Parameter(torch.randn(heads))
        self.dampen_factors = nn.Parameter(torch.randn(heads))

        if bidirectional:
            self.reverse_alphas = nn.Parameter(torch.randn(heads))
            self.reverse_dampen_factors = nn.Parameter(torch.randn(heads))

    def forward(self, x):
        device, seq_len = x.device, x.shape[1]

        # project in and split heads

        x = einsum('... d, h d -> ... h d', x, self.expansion)

        if self.bidirectional:
            x, x_reversed = x.chunk(2, dim = -2)
            x_reversed = torch.flip(x_reversed, dims = (1,))

        # weights derived from alphas (learned exponential smoothing decay rate)

        def apply_learned_ema_with_damping(x, alphas, dampen_factors):
            alphas = alphas.sigmoid()
            dampen_factors = dampen_factors.sigmoid()

            reversed_powers = torch.arange(seq_len - 1, -1, -1, device = device)
            K = alphas * (((1 - alphas) * dampen_factors) ** rearrange(reversed_powers, '... l -> ... l 1'))

            # conv1d fft O(nlog(n))

            return conv1d_fft(x, K, dim = -3, weight_dim = -2)

        x = apply_learned_ema_with_damping(x, self.alphas, self.dampen_factors)

        if self.bidirectional:
            x_reversed = apply_learned_ema_with_damping(x_reversed, self.reverse_alphas, self.reverse_dampen_factors)
            x_reversed = torch.flip(x_reversed, dims = (1,))
            x = torch.cat((x, x_reversed), dim = -2)

        # combine heads and out

        return einsum('... h d, h d -> ... d', x, self.reduction)

# Mega Layer
# Single headed Attention + Multi-headed EMA, then GRU-esque gating

class MegaLayer(nn.Module):
    def __init__(
        self,
        *,
        dim = 128,
        ema_heads = 16,
        attn_dim_qk = 64,
        attn_dim_value = 256,
        laplacian_attn_fn = False,
        causal = True,
        ema_dim_head = None
    ):
        super().__init__()

        self.single_headed_attn = SingleHeadedAttention(
            dim = dim,
            dim_qk = attn_dim_qk,
            dim_value = attn_dim_value,
            causal = causal,
            laplacian_attn_fn = laplacian_attn_fn
        )

        self.multi_headed_ema = MultiHeadedEMA(
            dim = dim,
            heads = ema_heads,
            bidirectional = not causal,
            dim_head = ema_dim_head
        )

        self.to_reset_gate = nn.Sequential(
            nn.Linear(dim, attn_dim_value),
            nn.SiLU()
        )

        self.to_update_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # equation 14, for calculating H

        self.Wh = nn.Parameter(torch.randn(dim, dim))
        self.Uh = nn.Parameter(torch.randn(attn_dim_value, dim))
        self.bh = nn.Parameter(torch.randn(dim))

    def forward(self, x, residual = None):
        residual = default(residual, x)

        ema_output = self.multi_headed_ema(x)
        attn_output = self.single_headed_attn(ema_output, x)

        reset_gate = self.to_reset_gate(ema_output)
        update_gate = self.to_update_gate(ema_output)

        gated_attn_output = attn_output * reset_gate

        # equation 14

        H = F.silu(ema_output @ self.Wh + gated_attn_output @ self.Uh + self.bh)

        # update gate

        return update_gate * H + (1 - update_gate) * x
    
    
    
    import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.fft import rfft, irfft

from einops import rearrange

# functions

def exists(val):
    return val is not None

# classes

class DSS(nn.Module):
    def __init__(
        self,
        *,
        dim,
        kernel_N = 512,
        dss_kernel_lambda_imag_exp = True
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # Lambda

        self.Lambda_real = nn.Parameter(torch.randn(kernel_N))
        self.Lambda_imag = nn.Parameter(torch.randn(kernel_N))

        # C

        self.C_real = nn.Parameter(torch.randn(dim, kernel_N))
        self.C_imag = nn.Parameter(torch.randn(dim, kernel_N))

        # params D

        self.param_D = nn.Parameter(torch.randn(dim))

        # whether to exponentiate lambda imag @albertfgu says it is not accurate to s4 original designs (but it is present in the pseudocode)

        self.dss_kernel_lambda_imag_exp = dss_kernel_lambda_imag_exp

    def forward(self, x):
        """
        einstein notation:
        b - batch
        l - sequence length
        d - dimension
        """

        device, seq_len = x.device, x.shape[1]
        u = self.norm(x)

        # learned weighted residual

        residual = u * self.param_D

        # derive simple dss kernel

        Lambda_imag = self.Lambda_imag.exp() if self.dss_kernel_lambda_imag_exp else self.Lambda_imag

        Lambda = -self.Lambda_real.exp() + 1j * Lambda_imag
        C = self.C_real + 1j * self.C_imag

        arange = torch.arange(seq_len, device = device)

        S = (rearrange(Lambda, 'n -> n 1') * rearrange(arange, 'l -> 1 l')).exp()
        C = C * (Lambda.exp() - 1) / Lambda

        K = einsum('h n, n l -> l h', C, S).real

        # conv1d fft O(nlog(n))

        u_f = rfft(u, n = seq_len * 2, dim = -2)
        K_f = rfft(K, n = seq_len * 2, dim = -2)

        y = irfft(u_f * K_f, seq_len * 2, dim = -2)[..., :seq_len, :]

        return y + residual

class GSS(nn.Module):
    """ Pseudocode 3.2 """

    def __init__(
        self,
        *,
        dim,
        dim_expansion_factor = 4,
        dss_kernel_N = 512,
        dss_kernel_H = 256,
        reverse_seq = False,
        dss_kernel_lambda_imag_exp = True
    ):
        super().__init__()
        self.reverse_seq = reverse_seq
        self.norm = nn.LayerNorm(dim)

        dim_hidden = int(dim_expansion_factor * dim)
        self.to_u = nn.Sequential(nn.Linear(dim, dim_hidden, bias = False), nn.GELU())
        self.to_v = nn.Sequential(nn.Linear(dim, dss_kernel_H, bias = False), nn.GELU())

        self.dss = DSS(dim = dss_kernel_H, kernel_N = dss_kernel_N, dss_kernel_lambda_imag_exp = dss_kernel_lambda_imag_exp)

        self.to_gate = nn.Linear(dss_kernel_H, dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        if self.reverse_seq:
            x = torch.flip(x, dims = (1,))

        residual, x = x.clone(), self.norm(x)

        u = self.to_u(x)
        v = self.to_v(x)

        v = self.dss(v)

        uc = self.to_gate(v)
        out = self.to_out(uc * u)

        out = out + residual

        if self.reverse_seq:
            out = torch.flip(out, dims = (1,))

        return 