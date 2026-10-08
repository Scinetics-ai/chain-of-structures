import re
L = "/data/caiqingran/enzymecot/runs/q35_deep2like_4stage_16g/train.log"
raw = open(L, errors="replace").read().split("\n")
last, out = None, []
for l in raw:
    m = re.match(r"step=(\d+) loss=", l)
    if m: last = int(m.group(1))
    if "MR-EVAL] val" in l and "s2=" in l:
        g = re.findall(r"s(\d)=([\d.]+|nan)/([\d.]+|nan)\(n=\d+\)", l)
        out.append((last, {int(k): (a, b) for k, a, b in g}))
D2 = {"终值@20950": ("0.7969","0.6984","0.8475","0.7529","0.7641","0.7512")}
print("%-8s %-10s %16s %16s %9s %9s" % ("步", "≈deep2步", "s2 结构/氨基酸", "s3 结构/氨基酸", "s4 结构", "s5 氨基酸"))
for s, d in out:
    print("%-8s %-10s %7s/%-8s %7s/%-8s %9s %9s"
          % (s, "~%d" % (s*4), d[2][0], d[2][1], d[3][0], d[3][1], d[4][0], d[5][1]))
v = D2["终值@20950"]
print("%-8s %-10s %7s/%-8s %7s/%-8s %9s %9s" % ("deep2", "终值", v[0], v[1], v[2], v[3], v[4], v[5]))
if out:
    s, d = out[-1]
    print("\n最新一次 vs deep2 终值:")
    for lab, mine, ref in (("s2 结构", d[2][0], v[0]), ("s2 氨基酸", d[2][1], v[1]),
                           ("s3 结构", d[3][0], v[2]), ("s3 氨基酸", d[3][1], v[3]),
                           ("s4 结构", d[4][0], v[4]), ("s5 氨基酸", d[5][1], v[5])):
        print("   %-10s %s vs %s  => %+.4f" % (lab, mine, ref, float(mine)-float(ref)))
