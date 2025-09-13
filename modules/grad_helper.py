import torch

class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_norm = torch.norm(grad_output)

        if grad_output_norm == 0:
            grad_output_normalized = grad_output
        else:
            grad_output_normalized = grad_output / grad_output_norm

        return grad_output_normalized