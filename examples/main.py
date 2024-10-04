from attention.impl import Attention, FlashAttention, FlashAttention_v2
from utils.diff import Diff

import torch
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attention implementation')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--q_head', type=int, default=1, help='head num of query')
    parser.add_argument('--kv_head', type=int, default=1, help='head num of key/value')
    parser.add_argument('--head_size', type=int, default=64, help='head size')
    parser.add_argument('--seq_q', type=int, default=512, help='sequence length of query')
    parser.add_argument('--seq_kv', type=int, default=512, help='sequence length of key/value')
    parser.add_argument('--softmax_scale', type=float, default=1, help='the scaling of qk^T before applying softmax')
    parser.add_argument('--causal', type=bool, default=False, help='whether to apply causal attention mask')

    args = parser.parse_args()
    query = torch.randn(size=(args.batch, args.seq_q, args.q_head, args.head_size))
    key = torch.randn(size=(args.batch, args.seq_kv, args.kv_head, args.head_size))
    value = torch.randn(size=(args.batch, args.seq_kv, args.kv_head, args.head_size))
    args.softmax_lse = 1.0 / np.sqrt(args.head_size)

    # base_output = torch.nn.functional.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), is_causal=args.causal, scale=args.softmax_scale)
    attn = FlashAttention(args.batch, args.q_head, args.kv_head, args.head_size, args.seq_q, args.seq_kv, args.softmax_scale, args.causal)
    base_output, base_lse = attn(query, key, value)
    attn = FlashAttention_v2(args.batch, args.q_head, args.kv_head, args.head_size, args.seq_q, args.seq_kv, args.softmax_scale, args.causal)
    real_output, real_lse = attn(query, key, value)

    differ = Diff()
    out_diff1, out_diff2 = differ(base_output, real_output)
    lse_diff1, lse_diff2 = differ(base_lse, real_lse)
    print(f'evaluation out diff1 is {out_diff1}, lse diff is {lse_diff1}')
    print(f'evaluation out diff2 is {out_diff2}, lse diff is {lse_diff2}')
