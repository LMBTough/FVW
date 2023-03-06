import torch
import numpy as np
from utils import get_device, parse_param
device = get_device()


def update_param(net, param, alpha, mask=None, op="add"):
    param = parse_param(param)
    if mask is not None:
        grad = np.array(eval(
            "net." + param + ".grad.cpu().detach().numpy()", {"net": net})) * mask.cpu().detach().numpy()
    else:
        grad = np.array(eval(
            "net." + param + ".grad.cpu().detach().numpy()", {"net": net}))
    weight_mask_add_1 = (eval(
        "(net." + param + ".cpu().detach().numpy())", {"net": net}) < 0) & (grad > 0)
    weight_mask_add_2 = (eval(
        "(net." + param + ".cpu().detach().numpy())", {"net": net}) > 0) & (grad < 0)
    weight_mask_add = weight_mask_add_1 ^ weight_mask_add_2
    weight_mask_minus = ~weight_mask_add
    if op == "add":
        weight = eval("net." + param + ".cpu().detach().numpy()", {"net": net}) + \
            alpha * np.sign(grad) * weight_mask_add
    elif op == "minus":
        weight = eval("net." + param + ".cpu().detach().numpy()", {"net": net}) - \
            alpha * np.sign(grad) * weight_mask_minus
    exec("net." + param + " = torch.nn.Parameter(torch.from_numpy(weight).to(device))",
         {"net": net, "device": get_device(), "weight": weight, "torch": torch})
    return grad


def forward_backward(net, train_loader, loss_func):
    total_loss = 0
    num = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        outputs = net(x)
        correct_index = outputs.argmax(-1) == y
        outputs = outputs[correct_index]
        y = y[correct_index]
        if len(outputs):
            loss = loss_func(outputs, y)
            total_loss = loss.item() + total_loss
            loss.backward()
            num += x.shape[0]
    return total_loss, num


def update_all_params(net, params, alp=0.000005, masks=None, op="add"):
    grads = list()
    for param in params:
        if masks is not None and param in masks:
            mask = masks[param]
        else:
            mask = None
        grad = update_param(net, param, alp, mask=mask, op=op)
        grads.append(grad)
    return grads


def update_totals(params, iters, totals, num, grads_before, grads, alp=0.000005, op="add",dataset="cifar10"):
    for param, grad_before, grad in zip(params, grads_before, grads):
        if totals[param] is None:
            if iters > 50 and op == "add" and dataset != "cifar100":
                totals[param] = (alp * np.sign(grad_before)) * \
                    (grad_before + grad) / num / 2
            else:
                totals[param] = -(alp * np.sign(grad_before)) * \
                    (grad_before + grad) / num / 2
        else:
            if iters > 50 and op == "add" and dataset != "cifar100":
                totals[param] += (alp * np.sign(grad_before)) * \
                    (grad_before + grad) / num / 2
            else:
                totals[param] += -(alp * np.sign(grad_before)) * \
                    (grad_before + grad) / num / 2
    return totals


def attack(net, iters, train_loader, params, masks=None, num_steps=5, alpha=0.000005, op="add",dataset="cifar10"):
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    totals = dict()
    for param in params:
        totals[param] = None
    start_loss = None
    end_loss = None
    grads_before = None
    grads = None
    for step in range(num_steps + 1):
        if masks is not None:
            assert isinstance(alpha, list)
            if step == 0:
                alp = alpha[0]
            else:
                alp = alpha[1]
        else:
            alp = alpha
        total_loss, num = forward_backward(
            net, train_loader, loss_func)
        if step == 0:
            start_loss = total_loss / num
            grads_before = update_all_params(
                net, params, alp=alp, masks=masks, op=op)
        else:
            grads = update_all_params(
                net, params, alp=alp, masks=masks, op=op)
            totals = update_totals(
                params, iters, totals, num, grads_before, grads, alp=alp, op=op,dataset=dataset)
            grads_before = grads
            end_loss = total_loss / num
        net.zero_grad()
    param_totals = list()
    for param in params:
        param_totals.append(totals[param])
    param_totals = np.array(param_totals, dtype=object)
    for param in params:
        totals[param] = param_totals[params.index(param)]
    return totals, end_loss - start_loss
