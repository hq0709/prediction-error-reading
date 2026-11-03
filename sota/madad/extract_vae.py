"""Extract MAD-AD's VAE (farzadbz/Medical-VAE, VAE-Medical-klf8.pt, MIT, ungated) without unpickling any code.

The released file is a whole pickled `ldm.models.autoencoder.AutoencoderKL` (a LightningModule that also carries its
LPIPS/discriminator loss), which MAD-AD loads with a plain `torch.load` (arbitrary code). Here it is read with
`torch.load(weights_only=True)`: every class the pickle names (listed with pickletools beforehand: ldm.*, taming.*,
torch.nn.modules.*) is mapped to an inert stub that only records its state, so no foreign constructor or __setstate__
runs. The encoder / decoder / quant_conv / post_quant_conv tensors and the architecture hyper-parameters are written out
as a plain state dict (loadable with weights_only=True) and a JSON config; vae.py rebuilds the model from the repo's own
ldm Encoder/Decoder code.

    python -I -B extract_vae.py /raid/home/CAMCA/hj880/ext_src/madad_vae/VAE-Medical-klf8.pt
"""
import collections
import json
import os
import pickletools
import sys
import zipfile

import torch

ALLOWED_NATIVE = {"collections OrderedDict", "torch._utils _rebuild_tensor_v2", "torch._utils _rebuild_parameter",
                  "torch FloatStorage", "torch LongStorage", "torch device", "torch float32", "__builtin__ set"}


class Stub:
    def __init__(self, *args, **kwargs):
        self._args, self._kwargs = args, kwargs

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


def pickle_globals(path):
    z = zipfile.ZipFile(path)
    pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
    return sorted({arg for op, arg, _ in pickletools.genops(z.read(pkl)) if op.name == "GLOBAL"})


def walk(m, prefix, out):
    d = m.__dict__
    for n, p in d.get("_parameters", {}).items():
        if p is not None:
            out[prefix + n] = p
    nonpersistent = d.get("_non_persistent_buffers_set", set())
    for n, b in d.get("_buffers", {}).items():
        if b is not None and n not in nonpersistent:
            out[prefix + n] = b
    for n, c in d.get("_modules", {}).items():
        if c is not None:
            walk(c, prefix + n + ".", out)


def main():
    src = sys.argv[1]
    out_dir = os.path.dirname(os.path.abspath(src))
    globs = pickle_globals(src)
    print("pickle globals:", globs)
    stubs = []
    for g in globs:
        if g in ALLOWED_NATIVE:
            continue
        mod, name = g.split(" ")
        stubs.append((type(name, (Stub,), {"_path": f"{mod}.{name}"}), f"{mod}.{name}"))
    with torch.serialization.safe_globals(stubs):
        root = torch.load(src, map_location="cpu", weights_only=True)
    print("root object:", type(root).__name__, getattr(type(root), "_path", None))
    sd = collections.OrderedDict()
    walk(root, "", sd)
    keep = collections.OrderedDict((k, v.detach().clone().float().contiguous()) for k, v in sd.items()
                                   if k.split(".")[0] in ("encoder", "decoder", "quant_conv", "post_quant_conv"))
    print(f"{len(sd)} tensors in the pickle, {len(keep)} kept (encoder/decoder/quant convs); dropped prefixes:",
          sorted({k.split('.')[0] for k in sd} - {"encoder", "decoder", "quant_conv", "post_quant_conv"}))
    enc, dec = root._modules["encoder"].__dict__, root._modules["decoder"].__dict__
    ch = enc["ch"]
    nres = enc["num_resolutions"]
    ch_mult = [keep[f"encoder.down.{i}.block.0.conv1.weight"].shape[0] // ch for i in range(nres)]
    attn_res, res = [], enc["resolution"]
    for i in range(nres):
        if len(enc["_modules"]["down"]._modules[str(i)]._modules["attn"]._modules):
            attn_res.append(res)
        if i != nres - 1:
            res //= 2
    cfg = {"embed_dim": root.__dict__["embed_dim"], "ch": ch, "ch_mult": ch_mult, "num_res_blocks": enc["num_res_blocks"],
           "attn_resolutions": attn_res, "in_channels": enc["in_channels"], "resolution": enc["resolution"],
           "z_channels": keep["encoder.conv_out.weight"].shape[0] // 2, "double_z": True,
           "out_ch": keep["decoder.conv_out.weight"].shape[0], "dropout": 0.0,
           "give_pre_end": dec.get("give_pre_end", False), "tanh_out": dec.get("tanh_out", False)}
    print("config:", cfg)
    torch.save(dict(keep), f"{out_dir}/klf8_vae_state.pt")
    json.dump(cfg, open(f"{out_dir}/klf8_vae_config.json", "w"), indent=1)
    print("wrote", f"{out_dir}/klf8_vae_state.pt", f"{out_dir}/klf8_vae_config.json")


if __name__ == "__main__":
    main()
