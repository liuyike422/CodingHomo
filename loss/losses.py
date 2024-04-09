import numpy as np
import torch

def compute_metrics(data, endpoints, manager):
    metrics = {}
    with torch.no_grad():
        # compute metrics
        B = data["label"].size()[0]
        outputs = np.argmax(endpoints["p"].detach().cpu().numpy(), axis=1)
        accuracy = np.sum(outputs.astype(np.int32) == data["label"].detach().cpu().numpy().astype(np.int32)) / B
        metrics['accuracy'] = accuracy
        return metrics


def ComputeErrH(src, dst, H):
    src_xy1 = torch.cat((src, src.new_ones(1)), -1).view(3, 1)
    src_d = torch.mm(H, src_xy1)
    small = 1e-7
    smallers = 1e-6 * (1.0 - torch.ge(torch.abs(src_d[-1]), small).float())
    src_d = src_d[:2] / (src_d[-1] + smallers)
    tmp_err = torch.norm(src_d - dst.view(-1, 1))
    return tmp_err


def ComputeErrFlow(src, dst, flow):
    src_t = src + flow[int(src[1]), int(src[0])]
    error = torch.linalg.norm(dst - src_t)
    return error


def ComputeErr(src, dst):
    error = torch.linalg.norm(dst - src)
    return error


def compute_eval_results(data_batch, output_batch):
    imgs_full = data_batch["imgs_gray_full"]

    device = imgs_full.device

    pt_set = list(map(eval, data_batch["pt_set"]))
    pt_set = list(map(lambda x: x['matche_pts'], pt_set))

    batch_size, _, img_h, img_w = imgs_full.shape
    flow_b = output_batch["flow_b"]#mv 为flow_b
    flow_f = output_batch["flow_f"]

    errs_m = []
    errs_i = []


    for i in range(batch_size):
        pts = torch.Tensor(pt_set[i]).to(device)
        err = 0
        ide = 0
        for j in range(6):
            p1 = pts[j][0]
            p2 = pts[j][1]
            src, dst = p1, p2
            err_b = ComputeErrFlow(src=dst, dst=src, flow=flow_b[i])
            err_f = ComputeErrFlow(src=src, dst=dst, flow=flow_f[i])
            err += min(err_b, err_f)
            ide += ComputeErr(src=src, dst=dst)

        err /= 6
        ide /= 6
        errs_m.append(err)
        errs_i.append(ide)

    eval_results = {"errors_m": errs_m, "errors_i": errs_i}#, 'rubost':rubost}

    return eval_results
