import torch

class Diff:
    def __init__(self, eps = 1e-7):
        self.eps = eps

    def compute_diff1(self, base, real):
        numerator = torch.sum(torch.abs(base - real))
        denominator = torch.sum(torch.abs(base))

        return numerator / (denominator + self.eps)

    def compute_diff2(self, base, real):
        numerator = torch.sum(torch.pow(base - real, 2))
        denominator = torch.sum(torch.pow(base, 2))

        return numerator / (denominator + self.eps)

    def __call__(self, base, real):
        diff1 = self.compute_diff1(base, real)
        diff2 = self.compute_diff2(base, real)

        return diff1, diff2
