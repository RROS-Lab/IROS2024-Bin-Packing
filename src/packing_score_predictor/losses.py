import torch

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t * torch.sigmoid(ey_t) - ey_t)
    


class MaxLoss(torch.nn.Module):
    def __init__(self):
        super(MaxLoss, self).__init__()
        
    def max_error(self, x, y):
        # x: [B, D]
        # y: [B, D]
        max_error = torch.max(torch.abs(x-y))


        return max_error

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.max_error(pred, label)
