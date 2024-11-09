import torch
from utils.quantize import absmax_quantize, zeropoint_quantize


a = torch.rand(size=(4,))
b = torch.randn(size=(4,))
print(f"original a is {a}")
print(f"original b is {b}")

a_am_quant, a_am_dequant = absmax_quantize(a)
b_am_quant, b_am_dequant = absmax_quantize(b)
print(f"[absmax] a quant is {a_am_quant}, dequant is {a_am_dequant}")
print(f"[absmax] b quant is {b_am_quant}, dequant is {b_am_dequant}")

a_zp_quant, a_zp_dequant = zeropoint_quantize(a)
b_zp_quant, b_zp_dequant = zeropoint_quantize(b)
print(f"[zeropoint] a quant is {a_zp_quant}, dequant is {a_zp_dequant}")
print(f"[zeropoint] b quant is {b_zp_quant}, dequant is {b_zp_dequant}")