from attention.impl import Attention, FlashAttention, FlashAttention_v2
from utils.diff import Diff
from decoder import Decoder

import torch
import argparse
import numpy as np

uniform = torch.distributions.uniform.Uniform(-0.1, 0.1)

def test_attention(batch, q_head, kv_head, head_size, seq_q, seq_kv, softmax_scale, causal):
    differ = Diff()
    query = uniform.sample(torch.Size([batch, seq_q, q_head, head_size]))
    key = uniform.sample(torch.Size([batch, seq_kv, kv_head, head_size]))
    value = uniform.sample(torch.Size([batch, seq_kv, kv_head, head_size]))

    attn = Attention(batch, q_head, kv_head, head_size, seq_q, seq_kv, softmax_scale, causal)
    base_output, base_lse = attn(query, key, value)

    attn = FlashAttention(batch, q_head, kv_head, head_size, seq_q, seq_kv, softmax_scale, causal)
    real_output, real_lse = attn(query, key, value)

    out_diff1, out_diff2 = differ(base_output, real_output)
    lse_diff1, lse_diff2 = differ(base_lse, real_lse)
    print(f'evaluation attn v1 out diff1 is {out_diff1}, lse diff is {lse_diff1}')
    print(f'evaluation attn v1 out diff2 is {out_diff2}, lse diff is {lse_diff2}')


    attn = FlashAttention_v2(batch, q_head, kv_head, head_size, seq_q, seq_kv, softmax_scale, causal)
    real_output, real_lse = attn(query, key, value)

    out_diff1, out_diff2 = differ(base_output, real_output)
    lse_diff1, lse_diff2 = differ(base_lse, real_lse)
    print(f'evaluation attn v2 out diff1 is {out_diff1}, lse diff is {lse_diff1}')
    print(f'evaluation attn v2 out diff2 is {out_diff2}, lse diff is {lse_diff2}')

def test_decoder(batch, q_head, head_size, seq_q, seq_out, softmax_scale, causal):
    inputs = uniform.sample(torch.Size([batch, seq_q, q_head * head_size]))
    weights = uniform.sample(torch.Size([q_head * head_size, 3 * q_head * head_size]))
    out_weights = uniform.sample(torch.Size([q_head * head_size, q_head * head_size]))

    decoder = Decoder(batch, q_head, head_size, seq_q, seq_out, softmax_scale, causal)
    base_output = decoder.decode(inputs, weights, out_weights)
    decoder = Decoder(batch, q_head, head_size, seq_q, seq_out, softmax_scale, causal)
    real_output = decoder(inputs, weights, out_weights)

    differ = Diff()
    out_diff1, out_diff2 = differ(base_output, real_output)
    print(f'evaluation decoder out diff1 is {out_diff1}')
    print(f'evaluation decoder out diff2 is {out_diff2}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attention implementation')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--q_head', type=int, default=1, help='head num of query')
    parser.add_argument('--kv_head', type=int, default=1, help='head num of key/value')
    parser.add_argument('--head_size', type=int, default=64, help='head size')
    parser.add_argument('--seq_q', type=int, default=512, help='sequence length of query')
    parser.add_argument('--seq_kv', type=int, default=512, help='sequence length of key/value')
    parser.add_argument('--seq_out', type=int, default=5, help='output sequence length for decoder')
    parser.add_argument('--softmax_scale', type=float, default=1, help='the scaling of qk^T before applying softmax')
    parser.add_argument('--causal', type=bool, default=False, help='whether to apply causal attention mask')

    args = parser.parse_args()
    args.softmax_lse = 1.0 / np.sqrt(args.head_size)

    test_attention(args.batch, args.q_head, args.kv_head, args.head_size, args.seq_q, args.seq_kv, args.softmax_scale, args.causal)
    test_decoder(args.batch, args.q_head, args.head_size, args.seq_q, args.seq_out, args.softmax_scale, args.causal)