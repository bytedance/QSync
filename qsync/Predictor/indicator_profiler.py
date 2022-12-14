import torch 
from qsync.LpTorch.conf import config
from qsync.LpTorch.quant import get_amax, get_scale
# profile
def profile_stas(q_input, q_weight, output, scale_inp, scale_w, id_layer):
    q_input, q_weight, output = q_input.cpu(), q_weight.cpu(), output.cpu()
    Dvv = D_O = output.nelement()
    with torch.no_grad():
        # if not config.training:
        # in run time, this two element needn't be calculated
        Dv, Dw = q_input.nelement(), q_weight.nelement() 
        if scale_inp is not None:
            # fixed point
            Sv, Sw = scale_inp, scale_w
            q_inp_norm2 = torch.norm(q_input * Sv).item()
            q_weight_norm2 = torch.norm(q_weight * Sw).item()
            
        else:
            # floating point
            q_inp_norm2 = torch.norm(q_input).item()
            q_weight_norm2 = torch.norm(q_weight).item()
            Sv, Sw = get_scale(get_amax(q_input)).item() , get_scale(get_amax(q_weight)).item() 
        if torch.is_tensor(Sv):
            Sv = torch.norm(Sv)
        elif torch.is_tensor(Sw):
            Sw = torch.norm(Sw)
        config.store_sta_act([Sv, Sw, Dv, Dw, Dvv, q_inp_norm2, q_weight_norm2], id_layer, tag='act')

def profile_stas_sole_input(q_input, scale_inp, id_layer):
    q_input = q_input.cpu()
    with torch.no_grad():
        # if not config.training:
        # in run time, this two element needn't be calculated
        Dvv = Dv = q_input.nelement()
        if scale_inp is not None:
            # fixed point
            Sv = scale_inp
            q_inp_norm2 = torch.norm(q_input * Sv).item()
        else:
            # floating point
            q_inp_norm2 = torch.norm(q_input).item()
            Sv = get_scale(get_amax(q_input)).item() 
        if torch.is_tensor(Sv):
            Sv = torch.norm(Sv)
        config.store_sta_act([Sv, 1, Dv, 1, Dvv, q_inp_norm2, 1], id_layer, tag='act')


def profile_stas_emb(q_input, scale_inp, output, id_layer):
    q_input = q_input.cpu()
    with torch.no_grad():
        # if not config.training:
        # in run time, this two element needn't be calculated
        Dv = q_input.nelement()
        Dvv = output.nelement()
        if scale_inp is not None:
            # fixed point
            Sv = scale_inp
            q_inp_norm2 = torch.norm(q_input * Sv).item()
        else:
            # floating point
            q_inp_norm2 = torch.norm(q_input).item()
            Sv = get_scale(get_amax(q_input)).item() 
        if torch.is_tensor(Sv):
            Sv = torch.norm(Sv)
        config.store_sta_act([Sv, 1, Dv, 1, Dvv, q_inp_norm2, 1], id_layer, tag='act')

