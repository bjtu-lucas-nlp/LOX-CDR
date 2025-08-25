import torch

class LossMAE(torch.nn.Module):
    def __init__(self):
        super(LossMAE, self).__init__()

    def forward(self, pred, real):
        # clone代表将pred 值复制给pred_loss
        pred_loss = pred.clone()
        # pred_loss在real中为0的位置上值置为0，因为0处的数据不算
        pred_loss[real == 0] = 0
        diffs = torch.add(real, -pred_loss)
        n = len(torch.nonzero(real))
        # mae = torch.sum(torch.abs(diffs)) / n
        mae = torch.sum(torch.abs(diffs))
        return mae


class LossMSE(torch.nn.Module):
    def __init__(self):
        super(LossMSE, self).__init__()

    def forward(self, pred, real):
        pred_loss = pred.clone()
        pred_loss[real == 0] = 0
        # print("lucas---real: ",real)
        # print("lucas---pred_loss: ",-pred_loss)
        diffs = torch.add(real, -pred_loss)
        n = len(torch.nonzero(real))
        # 不仅可以支持单个数运算也可以支持向量的运算
        # mse = torch.sum(diffs.pow(2)) / n
        mse = torch.sum(diffs.pow(2))
        return mse