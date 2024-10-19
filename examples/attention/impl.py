import torch
import numpy as np

LOG2E = np.log2(np.e)
MAX_THRESHOLD = 4.0

class Attention:
    def __init__(self, batch, q_head, kv_head, head_size, seq_q, seq_kv, softmax_scale, causal):
        self.bs = batch
        self.q_head = q_head
        self.kv_head = kv_head
        self.head_size = head_size
        self.seq_q = seq_q
        self.seq_kv = seq_kv
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.head_per_group = q_head // kv_head
        assert(q_head == self.head_per_group * kv_head)

    def __call__(self, query, key, value):
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.head_per_group > 1:
            query = query.reshape(self.bs, -1, self.head_per_group, self.seq_q, self.head_size)
            key = key.reshape(self.bs, self.kv_head, 1, self.seq_kv, self.head_size)
            value = value.reshape(self.bs, self.kv_head, 1, self.seq_kv, self.head_size)

        qk = torch.matmul(query, key.transpose(-1, -2)) * self.softmax_scale
        rmax = torch.amax(qk, dim=-1, keepdim=True)
        rsum = torch.sum(torch.exp(qk - rmax), dim=-1, keepdim=True)
        lse = torch.squeeze(rmax + torch.log(rsum))
        qk = torch.softmax(qk, dim=-1)
        qkv = torch.matmul(qk, value)

        if self.head_per_group > 1:
            qkv = qkv.reshape(self.bs, self.q_head, self.seq_q, self.head_size)
            lse = lse.reshape(self.bs, self.q_head, self.seq_q)
        qkv = qkv.transpose(1, 2)

        return qkv, lse

class FlashAttention(Attention):
    def __init__(self, batch, q_head, kv_head, head_size, seq_q, seq_kv, softmax_scale, causal):
        super().__init__(batch, q_head, kv_head, head_size, seq_q, seq_kv, softmax_scale, causal)
        self.block_q = 64
        self.block_kv = 512

    def compute_block(self, q, k, v, qkv, rmax, rsum, j):
        qk = torch.matmul(q, k.t()) * self.softmax_scale * LOG2E
        cmax = torch.amax(qk, dim=-1, keepdim=True)
        cmax = torch.maximum(cmax, rmax)
        qk = torch.pow(2, qk - cmax)
        scale = torch.pow(2, rmax - cmax)
        csum = torch.sum(qk, dim=-1, keepdim=True)
        csum = rsum * scale + csum
        qkv = qkv * scale + torch.matmul(qk, v)

        return qkv, cmax, csum

    def __call__(self, query, key, value):
        loop_q = (self.seq_q + self.block_q - 1) // self.block_q
        loop_kv = (self.seq_kv + self.block_kv - 1) // self.block_kv
        output = torch.empty(size=(self.bs, self.seq_q, self.q_head, self.head_size))
        lse = torch.empty(size=(self.bs, self.q_head, self.seq_q))
        for b in range(self.bs):
            for h in range(self.q_head):
                for i in range(loop_q):
                    sq = min(self.block_q, self.seq_q - i * self.block_q)
                    q = query[b, i * self.block_q : i * self.block_q + sq, h]
                    qkv = torch.zeros_like(q)
                    rmax = torch.ones(size=(sq, 1)) * float('-inf')
                    rsum = torch.zeros_like(rmax)
                    for j in range(loop_kv):
                        sk = min(self.block_kv, self.seq_kv - j * self.block_kv)
                        k = key[b, j * self.block_kv : j * self.block_kv + sk, h // self.head_per_group]
                        v = value[b, j * self.block_kv : j * self.block_kv + sk, h // self.head_per_group]
                        qkv, rmax, rsum = self.compute_block(q, k, v, qkv, rmax, rsum, j)
                    output[b, i * self.block_q : i * self.block_q + sq, h] = qkv / rsum
                    lse[b, h, i * self.block_q : i * self.block_q + sq] = torch.squeeze(rmax / LOG2E + torch.log(rsum))

        return output, lse

class FlashAttention_v2(FlashAttention):
    def __init__(self, batch, q_head, kv_head, head_size, seq_q, seq_kv, softmax_scale, causal):
        super().__init__(batch, q_head, kv_head, head_size, seq_q, seq_kv, softmax_scale, causal)

    def compute_block(self, q, k, v, qkv, rmax, rsum, j):
        if j == 0:
            qk = qk = torch.matmul(q, k.t()) * self.softmax_scale * LOG2E
        else:
            qk = qk = torch.matmul(q, k.t()) * self.softmax_scale * LOG2E - rmax

        cmax = torch.amax(qk, dim=-1, keepdim=True)
        max_value = torch.amax(cmax)

        if j == 0:
            qk = qk - cmax
            qk = torch.pow(2, qk)
            rsum = torch.sum(qk, dim=-1, keepdim=True)
            qkv = torch.matmul(qk, v)
            rmax = cmax
        elif max_value > MAX_THRESHOLD:
            qk = qk - cmax
            cmax = cmax + rmax
            qk = torch.pow(2, qk)
            csum = torch.sum(qk, dim=-1, keepdim=True)
            rscale = torch.pow(2, rmax - cmax)
            rsum = rsum * rscale + csum
            qkv = qkv * rscale + torch.matmul(qk, v)
            rmax = torch.maximum(cmax, rmax)
        else:
            qk = torch.pow(2, qk)
            csum = torch.sum(qk, dim=-1, keepdim=True)
            rsum = rsum + csum
            qkv = qkv + torch.matmul(qk, v)

        return qkv, rmax, rsum

    def __call__(self, query, key, value):
        return super().__call__(query, key, value)
