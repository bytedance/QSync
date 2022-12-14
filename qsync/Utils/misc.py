import torch 
def inp_to_device(inp, device):
    if isinstance(inp, dict):
        for k, v in inp.items():
            inp[k] = v.to(device)
    elif isinstance(inp, list):
        for ii, v in enumerate(inp):
            inp[ii] = v.to(device)
    elif isinstance(inp, torch.Tensor):
        inp = inp.to(device)
    else:
        print("warning: tried to move un-handled input to some device")
    return inp
