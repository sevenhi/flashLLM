from attention.impl import FlashAttention_v2

import torch



class Decoder:
    def __init__(self, batch, head_num, head_size, seq_input, seq_output, softmax_scale, causal):
        self.batch = batch
        self.head_num = head_num
        self.head_size = head_size
        self.seq_input = seq_input
        self.seq_output = seq_output
        self.causal = causal
        self.softmax_scale = softmax_scale

    def step(self, inputs, weights, out_weights):
        inputs = torch.matmul(inputs, weights)
        query, key, value = torch.split(inputs, self.head_num * self.head_size, dim=-1)
        query = query.reshape(self.batch, self.seq_input, self.head_num, self.head_size)
        key = key.reshape(self.batch, self.seq_input, self.head_num, self.head_size)
        value = value.reshape(self.batch, self.seq_input, self.head_num, self.head_size)
        attn = FlashAttention_v2(self.batch, self.head_num, self.head_num, self.head_size, self.seq_input, self.seq_input, self.softmax_scale, self.causal)
        output, lse = attn(query, key, value)
        output = output.reshape(self.batch, self.seq_input, self.head_num * self.head_size)
        output = torch.matmul(output, out_weights)

        return output[:, -1:, ...]

    def step_with_kvcache(self, inputs, weights, out_weights, k_cache, v_cache):
        inputs = torch.matmul(inputs, weights)
        query, key, value = torch.split(inputs, self.head_num * self.head_size, dim=-1)
        query = query.reshape(self.batch, 1, self.head_num, self.head_size)
        key = key.reshape(self.batch, 1, self.head_num, self.head_size)
        value = value.reshape(self.batch, 1, self.head_num, self.head_size)
        key = torch.cat((k_cache, key), dim=1)
        value = torch.cat((v_cache, value), dim=1)
        attn = FlashAttention_v2(self.batch, self.head_num, self.head_num, self.head_size, 1, self.seq_input, self.softmax_scale, self.causal)
        output, lse = attn(query, key, value)
        output = output.reshape(self.batch, 1, self.head_num * self.head_size)
        output = torch.matmul(output, out_weights)

        return output, key, value

    def decode(self, inputs, weights, out_weights):
        for i in range(self.seq_output):
            out = self.step(inputs, weights, out_weights)
            inputs = torch.cat((inputs, out), dim=1)
            self.seq_input += 1

        return inputs[:, -self.seq_output:, ...]

    def decode_with_kvcache(self, inputs, weights, out_weights):
        temp = torch.matmul(inputs[:, :-1, ...], weights)
        query, key, value = torch.split(temp, self.head_num * self.head_size, dim=-1)
        k_cache = key.reshape(self.batch, -1, self.head_num, self.head_size)
        v_cache = value.reshape(self.batch, -1, self.head_num, self.head_size)

        for i in range(self.seq_output):
            out, k_cache, v_cache = self.step_with_kvcache(inputs[:, -1:, ...], weights, out_weights, k_cache, v_cache)
            inputs = torch.cat((inputs, out), dim=1)
            self.seq_input += 1

        return inputs[:, -self.seq_output:, ...]

    def __call__(self, inputs, weights, out_weights):
        return self.decode_with_kvcache(inputs, weights, out_weights)
