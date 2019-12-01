# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/11/30 下午11:00
"""

if __name__ == "__main__":
    layers = nn.ModuleList([
        DilationBlockV1(1, residual_channels=12, kernel_size=2, dilation=1),
        DilationBlockV1(12, residual_channels=12, kernel_size=2, dilation=2),
        DilationBlockV1(12, residual_channels=12, kernel_size=2, dilation=4),
        DilationBlockV1(12, residual_channels=12, kernel_size=2, dilation=8),
        DilationBlockV1(12, residual_channels=12, kernel_size=2, dilation=16),
    ])
    from torch.nn import init
    # layer = DilationBlockV1(12, residual_channels=12, kernel_size=2, dilation=16)
    x = torch.normal(0, 0.05, (1, 1, 196))
    skips = torch.zeros((1, 12, 196))

    conv_out = nn.Conv1d(12, 1, kernel_size=1)

    print(x.shape)

    x_in = x
    for layer in layers:
        x_in, skip = layer(x_in)
        skips += skip
        print(x_in.shape, skips.shape)

    y = conv_out(skips)
    print(y.shape)
    import matplotlib.pyplot as plt

    plt.plot(x.detach().numpy().reshape(-1), label='x')
    plt.plot(y.detach().numpy().reshape(-1), label='y')
    plt.legend()
    plt.show()