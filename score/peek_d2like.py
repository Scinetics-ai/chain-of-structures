import torch, collections
p = "/data/caiqingran/enzymecot/runs/q35_deep2like_4stage_16g/FINAL_d2like_step9000_val0.7807.pt"
d = torch.load(p, map_location="cpu", weights_only=False)
print("  顶层类型:", type(d).__name__)
if isinstance(d, dict):
    print("  顶层键:", [k for k in list(d)[:14]])
    for k in list(d):
        v = d[k]
        if isinstance(v, dict) and v and all(hasattr(x, "shape") for x in list(v.values())[:3]):
            n = sum(t.numel() for t in v.values() if hasattr(t, "numel"))
            pref = collections.Counter(kk.split(".")[0] for kk in v)
            print("    %-22s 张量 %5d 个, 参数 %.1fM" % (k, len(v), n/1e6))
            for pp, c in pref.most_common(8):
                sub = sum(t.numel() for kk, t in v.items() if kk.split(".")[0] == pp and hasattr(t, "numel"))
                print("        %-28s %4d 个  %7.1fM" % (pp, c, sub/1e6))
        elif not isinstance(v, dict):
            print("    %-22s %s" % (k, str(v)[:70]))
