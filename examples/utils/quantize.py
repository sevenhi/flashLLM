import torch

def absmax_quantize(x):
    '''
    symmetric quantization
    '''
    scale = 127 / torch.max(torch.abs(x))
    quant = torch.round(scale * x)
    dequant = quant / scale

    return quant.to(torch.int8), dequant

def zeropoint_quantize(x):
    '''
    asymmetric quantization
    '''
    span = torch.max(x) - torch.min(x)
    span = 1 if span == 0 else span
    scale = 255 / span
    shift = torch.round(-scale * torch.min(x) - 128)
    quant = torch.clip(torch.round(scale * x + shift), -128, 127)
    dequant = (quant - shift) / scale

    return quant.to(torch.int8), dequant