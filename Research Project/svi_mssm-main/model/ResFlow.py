import torch
from torch import nn
from Lipschitz import LipschitzCNN1D
from IresBlock import iResBlock
from Other_Flow import EXP_Flow

class NormalizingFlow(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det += log_d
        return z, log_det


class Resflow_Each_Cluster(nn.Module):
    def __init__(self, base_param_dict, resflow_param_dict):
        super().__init__()
        num_flow_module = resflow_param_dict["num_flow_module"]
        dims = resflow_param_dict["dims"]
        kernel_size = resflow_param_dict["kernel_size"]
        self.mode = base_param_dict["mode"]

        kernel_size_list = [kernel_size] * (len(dims) - 1)
        flows = []
        for i in range(num_flow_module):
            net = LipschitzCNN1D(
                channels=dims,
                kernel_size=kernel_size_list,
                lipschitz_const=resflow_param_dict["coeff"],
                max_lipschitz_iter=resflow_param_dict["n_iterations"],
                lipschitz_tolerance=resflow_param_dict["tolerance"]
            )
            flows.append(iResBlock(
                net=net,
                geom_p=0.5,
                n_samples=1,
                n_exact_terms=2,
                neumann_grad=resflow_param_dict["reduce_memory"],
                grad_in_forward=resflow_param_dict["reduce_memory"]
            ))
        self.cnn_res = NormalizingFlow(flows=flows)

        if self.mode == "sir":
            self.exp_flow = EXP_Flow()

    def forward(self, xi):
        xi = xi.permute(0, 2, 1)
        z, log_det = self.cnn_res(xi)
        z = z.permute(0, 2, 1)

        if self.mode == "sl":
            return {"z": z, "log_det": log_det}
        elif self.mode == "sir":
            z, log_det_exp = self.exp_flow(z)
            log_det += log_det_exp
            z_norm = z / (z.sum(dim=2, keepdim=True))
            return {"z": z, "log_det": log_det, "z_norm": z_norm}


class Resflow_Multi_Cluster(nn.Module):
    def __init__(self, base_param_dict, resflow_param_dict):
        super().__init__()
        self.mode = base_param_dict["mode"]
        self.flow_models = nn.ModuleList()

    def forward(self, x, cluster_weights):
        z_list, log_det_list = [], []

        for weight in cluster_weights.T:
            flow_output = Resflow_Each_Cluster(base_param_dict, resflow_param_dict).forward(x * weight)
            z_list.append(flow_output["z"])
            log_det_list.append(flow_output["log_det"])

        if self.mode == "sl":
            return {"z": torch.stack(z_list, dim=1), "log_det": torch.stack(log_det_list, dim=1)}
        elif self.mode == "sir":
            z_norm_list = [output["z_norm"] for output in z_list]
            return {"z": torch.stack(z_list, dim=1), 
                    "log_det": torch.stack(log_det_list, dim=1), 
                    "z_norm": torch.stack(z_norm_list, dim=1)}
