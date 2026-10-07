# -*- coding: utf-8 -*-
"""机械执行预先写死的停训规则, 不凭感觉。

★规则(训练开始前定的, 见 README.zh.md):
  决策只看 `val_overlap` 的 R1 答案 CE。连续 2 个 val 点没有新低 ⇒ 停, 取 CE 最低的检查点。
  `val_strict` 照样打印但不进决策(它的表覆盖低 10 分, CE 不跨覆盖率可比 ——
  feedback_ce_is_not_comparable_across_coverage)。
★为什么要脚本: v11 我凭"看起来平了"说停滞, 第 4 个点出来又得撤回
  (feedback_per_step_loss_is_not_a_trend)。规则交给代码执行就不会摇摆。
用法: apply_stop_rule.py <train_node0.log>
"""
import re
import sys

log = open(sys.argv[1], errors="ignore").read().split("\n")
step = None
rows = []
for ln in log:
    m = re.match(r"\s+步 (\d+)/(\d+)", ln)
    if m:
        step, total = int(m.group(1)), int(m.group(2))
    for tag in ("val_overlap", "val_strict"):
        if f"[{tag}] ★答案" in ln:
            acc = dict((k, v) for k, v in re.findall(r"(R[1-4]) CE [0-9.]+/准确 ([0-9.]+)%", ln))
            rows.append({"step": step, "tag": tag,
                         "ce": {k: float(v) for k, v in
                                re.findall(r"(R[1-4]) CE ([0-9.]+)/", ln)},
                         "acc": {k: float(v) for k, v in acc.items()}})
ov = [r for r in rows if r["tag"] == "val_overlap"]
st = [r for r in rows if r["tag"] == "val_strict"]
if not ov:
    print("  还没有 val_overlap 的读数")
    sys.exit(0)
print(f"  {'步':>6}  {'R1 CE':>7}{'R1准确':>8}  {'R2 CE':>7}  {'R3 CE':>7}  {'R4 CE':>7}   新低?")
best = None
nolow = 0
hist = []
for r in ov:
    c = r["ce"].get("R1")
    low = best is None or c < best - 1e-9
    if low:
        best = c
        nolow = 0
    else:
        nolow += 1
    hist.append((r["step"], c, low, nolow))
    print(f"  {r['step']:>6}  {c:7.4f}{r['acc'].get('R1',0):7.1f}%  "
          f"{r['ce'].get('R2',0):7.4f}  {r['ce'].get('R3',0):7.4f}  {r['ce'].get('R4',0):7.4f}"
          f"   {'★新低' if low else f'否({nolow})'}")
print(f"\n  最低 R1 CE = {best:.4f} @ 步 {[h[0] for h in hist if h[1]==best][0]}")
print(f"  连续无新低次数 = {nolow}  (规则: 到 2 就停)")
if nolow >= 2:
    print(f"\n  ⇒ **按规则该停了**。取步 {[h[0] for h in hist if h[1]==best][0]} 附近的检查点。")
else:
    print(f"\n  ⇒ 继续训(还差 {2-nolow} 个无新低的点)。")
if st:
    print(f"\n  val_strict(只做读数, 不进决策; 它表覆盖 51.5% vs overlap 61.0%, CE 不可跨比):")
    for r in st[-3:]:
        print(f"    步 {r['step']:>5}  R1 CE {r['ce'].get('R1',0):.4f}/"
              f"{r['acc'].get('R1',0):.1f}%  R2 {r['ce'].get('R2',0):.4f}  "
              f"R3 {r['ce'].get('R3',0):.4f}  R4 {r['ce'].get('R4',0):.4f}")
print("\n  ★提醒: CE 只决定停在哪一步。能不能用要看离线配对评测打不打赢查表-only")
print("    (test_overlap 整对 23.4% / 氨基酸@选对 67.4%, 按 tplD 分层看)。")
