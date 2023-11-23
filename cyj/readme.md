# 进度

吴恩达那个能跑，但是其实没啥用，因为是用gym跑的，我试过把gym改成gymnasium来import highway环境，但是会有一些矩阵格式上面的错误，即在

    # model.py
    state = torch.tensor([state], dtype=torch.float).to(self.device)

这一行报错，原因是矩阵维度和expected维度不相符

并且这个错误不是因为highway和pendulum的环境问题，而是因为gym和gymnasium的问题，因为我在gymnasium中尝试使用pendulum环境也会报相同的错误

我主要是看他这个显示做的挺好的，用tqdm，又记录了loss啥的就想能不能搬迁过来，但是估计是不行，我之后会尝试cleanRL里面SAC的跑法，看看搞懂它的代码以及多增加一些显示。