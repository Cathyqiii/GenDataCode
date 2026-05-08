'''
Always evaluate the model with MAE, RMSE, MAPE, RRSE, PNBI, and oPNBI.
Why add mask to MAE and RMSE?
    Filter the 0 that may be caused by error (such as loop sensor)
Why add mask to MAPE and MARE?
    Ignore very small values (e.g., 0.5/0.5=100%)
'''
import numpy as np
import torch


def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))


def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)


def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def RRSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))


def CORR_torch(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(dim=1)
        true = true.transpose(1, 2).unsqueeze(dim=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(dim=dims)
    true_mean = true.mean(dim=dims)
    pred_std = pred.std(dim=dims)
    true_std = true.std(dim=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(dim=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation


# def MAPE_torch(pred, true, mask_value=None):
#     if mask_value != None:
#         mask = torch.gt(true, mask_value)
#         pred = torch.masked_select(pred, mask)
#         true = torch.masked_select(true, mask)
#     return torch.mean(torch.abs(torch.div((true - pred), true)))

def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    # 核心新增：把true中的0值替换为1e-8，避免除0爆炸
    true_safe = torch.where(true == 0, torch.tensor(1e-8, device=true.device), true)
    return torch.mean(torch.abs(torch.div((true - pred), true_safe)))


def PNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    indicator = torch.gt(pred - true, 0).float()
    return indicator.mean()


# def oPNBI_torch(pred, true, mask_value=None):
#     if mask_value != None:
#         mask = torch.gt(true, mask_value)
#         pred = torch.masked_select(pred, mask)
#         true = torch.masked_select(true, mask)
#     bias = (true+pred) / (2*true)
#     return bias.mean()
def oPNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    # 核心新增：避免true=0导致除0（和oPNBI_np逻辑一致）
    true_safe = torch.where(true == 0, torch.tensor(1e-8, device=true.device), true)
    bias = (true+pred) / (2*true_safe)
    return bias.mean()


def MARE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.div(torch.sum(torch.abs((true - pred))), torch.sum(true))


def SMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred)/(torch.abs(true)+torch.abs(pred)))


def MAE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE


def RMSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE


#Root Relative Squared Error
def RRSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    mean = true.mean()
    return np.divide(np.sqrt(np.sum((pred-true) ** 2)), np.sqrt(np.sum((true-mean) ** 2)))


# def MAPE_np(pred, true, mask_value=None):
#     if mask_value != None:
#         mask = np.where(true > (mask_value), True, False)
#         true = true[mask]
#         pred = pred[mask]
#     return np.mean(np.absolute(np.divide((true - pred), true)))

def MAPE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    # 核心新增：numpy版本也加0值兜底，避免除0爆炸
    true_safe = np.where(true == 0, 1e-8, true)
    return np.mean(np.absolute(np.divide((true - pred), true_safe)))


def PNBI_np(pred, true, mask_value=None):
    #if PNBI=0, all pred are smaller than true
    #if PNBI=1, all pred are bigger than true
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = pred-true
    indicator = np.where(bias>0, True, False)
    return indicator.mean()


# def oPNBI_np(pred, true, mask_value=None):
#     #if oPNBI>1, pred are bigger than true
#     #if oPNBI<1, pred are smaller than true
#     #however, this metric is too sentive to small values. Not good!
#     if mask_value != None:
#         mask = np.where(true > (mask_value), True, False)
#         true = true[mask]
#         pred = pred[mask]
#     bias = (true + pred) / (2 * true)
#     return bias.mean()

def oPNBI_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    # 核心新增：避免true=0导致除0
    true_safe = np.where(true == 0, 1e-8, true)
    bias = (true + pred) / (2 * true_safe)
    return bias.mean()


def MARE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true> (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.divide(np.sum(np.absolute((true - pred))), np.sum(true))


def CORR_np(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        #B, N
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        #np.transpose include permute, B, T, N
        pred = np.expand_dims(pred.transpose(0, 2, 1), axis=1)
        true = np.expand_dims(true.transpose(0, 2, 1), axis=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(0, 1, 2, 3)
        true = true.transpose(0, 1, 2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(axis=dims)
    true_mean = true.mean(axis=dims)
    pred_std = pred.std(axis=dims)
    true_std = true.std(axis=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(axis=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation


def All_Metrics(pred, true, mask1, mask2):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = MAE_np(pred, true, mask1)
        rmse = RMSE_np(pred, true, mask1)
        mape = MAPE_np(pred, true, mask2)
        #rrse = RRSE_np(pred, true, mask1)
        #corr = 0
        #corr = CORR_np(pred, true, mask1)
        #pnbi = PNBI_np(pred, true, mask1)
        #opnbi = oPNBI_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        mape = MAPE_torch(pred, true, mask2)
        #rrse = RRSE_torch(pred, true, mask1)
        #corr = CORR_torch(pred, true, mask1)
        #pnbi = PNBI_torch(pred, true, mask1)
        #opnbi = oPNBI_torch(pred, true, mask2)
    else:
        raise TypeError
    return mae, rmse, mape

# def All_Metrics(y_pred, y_true, mae_thresh=0, mape_thresh=1):
#     """
#     mape_thresh: 真实值小于这个数的样本，不参与 MAPE 计算
#     你可以根据数据集自己调：
#     - AirQuality: 0.01~0.05
#     - ETT / FD001: 0.1
#     - Traffic: 1.0
#     """
#     # 只保留 y_true > mape_thresh 的样本，防止分母接近0
#     mask = y_true > mape_thresh
#
#     y_pred_valid = y_pred[mask]
#     y_true_valid = y_true[mask]
#
#     # 剩下正常的点再算指标
#     mae = torch.mean(torch.abs(y_pred_valid - y_true_valid))
#     rmse = torch.sqrt(torch.mean((y_pred_valid - y_true_valid) ** 2))
#     mape = torch.mean(torch.abs((y_true_valid - y_pred_valid) / y_true_valid))
#
#     return mae, rmse, mape


def SIGIR_Metrics(pred, true, mask1, mask2):
    rrse = RRSE_torch(pred, true, mask1)
    corr = CORR_torch(pred, true, 0)
    return rrse, corr


if __name__ == '__main__':
    pred = torch.Tensor([1, 2, 3, 4])
    true = torch.Tensor([2, 1, 4, 5])
    print(All_Metrics(pred, true, None, None))

