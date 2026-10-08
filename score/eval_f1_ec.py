# -*- coding: utf-8 -*-
"""按 CFP-Gen 官方 eval/eval_ec.py 的口径算 mic-F1 / mac-F1 / AUPR / AUC。

★为什么照官方而不自己造: 2026-09-20 我们用这套算 CFP-Gen 复现出
  mac-F1 0.671(论文 0.673) / mic-F1 0.725(论文 0.722) —— 这把尺已经验过, 可信。
  自己换口径就失去了那次校准。

★官方算法里四个必须照搬的细节(逐行对齐 _archive_notes_20260729/cfpgen/eval/eval_ec.py):
  1 阈值: CLEAN 输出是 `EC:x.x.x.x/score`, 只保留 **score > 0.01** 的预测
    (`safe_split_and_check(threshold=0.01)`)
  2 标签空间: MultiLabelBinarizer **只 fit 在 y_true 上**, 预测里没出现在 y_true 的 EC
    被 transform 直接丢掉 —— 所以假阳不会无限惩罚
  3 层级: **四级全串**(1.1.1.1), 不截到三级。这和我们 band_final.py 的 ec3() 不同,
    那个截三级且只问交集非空 ⇒ 两套口径必须分开报, 不能混
  4 AUC/AUPR 用**二值预测**算, 不是概率 ⇒ 数值偏低且与 F1 强相关, 不要当概率型 AUC 读

★另外报我们自己的 EC3(截三级, 交集非空), 这样和已引用的 57.0 能对上。
用法: eval_f1_ec.py <maxsep.csv...> --labels <gt.json>
"""
import argparse
import json
import re
from collections import Counter


def clean_preds(parts, threshold=0.01):
    """照 safe_split_and_check: 只要 `EC:x/score` 且 score>threshold 的。"""
    out = set()
    for x in parts:
        x = x.strip()
        if not x or "/" not in x:
            continue
        a, _, b = x.rpartition("/")
        try:
            if float(b) <= threshold:
                continue
        except ValueError:
            continue
        a = a.replace("EC:", "").strip()
        if a:
            out.add(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--labels", required=True, help="idx -> [EC 四级字符串] 的 json")
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--subset", default="", help="只算这些 idx(一行一个)。"
                    "★必须用: 已引用的 57.0 是在【与基线配对的 2,435 条子集】上测的, "
                    "不是全部 2,801 条。少的 366 条是被长度过滤掉的长蛋白, 更难, "
                    "全集上 EC3 会低约 8 分。混用会让表里的数互相矛盾。")
    a = ap.parse_args()

    lab = json.load(open(a.labels))
    keep = None
    if a.subset:
        keep = {x.strip() for x in open(a.subset) if x.strip()}
        print(f"  子集限定 {len(keep):,} 条")
    y_true, y_pred, miss = [], [], 0
    ec3_ok = ec3_n = 0
    for p in a.csvs:
        for ln in open(p):
            ln = ln.strip()
            if not ln:
                continue
            head, *parts = ln.split(",")
            _, _, idx = head.partition("|")
            idx = (idx or head).strip()
            if keep is not None and idx not in keep:
                continue
            want = lab.get(idx)
            if not want:
                miss += 1
                continue
            got = clean_preds(parts, a.threshold)
            y_true.append(set(want))
            y_pred.append(got)
            # 我们自己的 EC3 口径: 截三级, 交集非空
            w3 = {".".join(x.split(".")[:3]) for x in want}
            g3 = {".".join(x.split(".")[:3]) for x in got}
            ec3_n += 1
            ec3_ok += bool(w3 & g3)

    print(f"  记录 {len(y_true):,} 条 | 标签表里查不到的 {miss} 条")
    assert len(y_true) > 0, "★一条都没对上, 检查 idx 格式"
    # ★join 命中率必须断言(feedback_join_key_must_assert_hit_rate)
    hit = len(y_true) / max(len(y_true) + miss, 1)
    assert hit > 0.95, f"★标签 join 命中率只有 {hit:.1%}, idx 格式可能不对"

    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    Yt = mlb.fit_transform(y_true)          # ★只 fit 在 y_true 上, 照官方
    Yp = mlb.transform([{x for x in s if x in set(mlb.classes_)} for s in y_pred])
    n_cls = Yt.shape[1]
    kept = sum(len({x for x in s if x in set(mlb.classes_)}) for s in y_pred)
    raw = sum(len(s) for s in y_pred)
    print(f"  标签空间 {n_cls} 个 EC(四级, 只来自真值) | 预测保留 {kept:,}/{raw:,} "
          f"= {kept/max(raw,1):.1%} (其余不在真值标签空间里, 照官方丢弃)")

    f = lambda fn, av: fn(Yt, Yp, average=av, zero_division=0)
    print()
    print(f"  {'指标':<22} {'Macro':>8} {'Micro':>8}")
    print(f"  {'F1':<22} {f(f1_score,'macro'):8.3f} {f(f1_score,'micro'):8.3f}")
    print(f"  {'Precision':<22} {f(precision_score,'macro'):8.3f} {f(precision_score,'micro'):8.3f}")
    print(f"  {'Recall':<22} {f(recall_score,'macro'):8.3f} {f(recall_score,'micro'):8.3f}")
    try:
        print(f"  {'AUPR (二值预测)':<22} {average_precision_score(Yt,Yp,average='macro'):8.3f} "
              f"{average_precision_score(Yt,Yp,average='micro'):8.3f}")
        print(f"  {'AUC-ROC (二值预测)':<22} {roc_auc_score(Yt,Yp,average='macro'):8.3f} "
              f"{roc_auc_score(Yt,Yp,average='micro'):8.3f}")
    except ValueError as e:
        print(f"  AUPR/AUC 算不了: {e}")
    print()
    print(f"  ★我们自己的 EC3 口径(截三级, 交集非空): {100.0*ec3_ok/ec3_n:.1f}% ({ec3_ok}/{ec3_n})")
    print("    ↑ 这个数应与已引用的 57.0 一致; 不一致说明记录或口径对不上, 必须查清")
    print("  参照(同一套官方口径, 2026-09-20): CFP-Gen 复现 mac-F1 0.671 / mic-F1 0.725")


if __name__ == "__main__":
    main()
