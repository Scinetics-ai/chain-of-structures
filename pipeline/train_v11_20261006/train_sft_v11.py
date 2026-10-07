# -*- coding: utf-8 -*-
"""v11 四轮阶梯的 SFT: Qwen3.5-9B + LoRA, 答案 token 加权, 16 卡两机 DDP。

★为什么 LoRA 不是全量(project_full_ft_destroys_qwen_lora_does_not):
  全量微调之后问"法国首都"会吐 `L A H Q L R S T T T V Y Y Y` —— 根因是输出 100% 是氨基酸,
  不是参数量。LoRA 不会。
★为什么整段 assistant 都监督、只给答案 token 加权(answer_token_weights.py 里有完整论证):
  只监督答案 ⇒ reasoning 没人管会吐垃圾, 而答案条件在垃圾上; 我们自己两次把东西从损失里
  拿掉都毁了能力(copy_weight_zero_destroys_the_read_path 97.15%→17.71%;
  stage2_rounds_regressed_to_the_prior)。实测 W=6 时答案拿到 22.7% 的梯度。
★按轮分开记 loss: 四轮难度差很多(R1 没有候选表、token 数只有 R4 的 1/4),
  总体平均会把它们盖掉(feedback_mixed_bucket_metric_hides_collapse)。

★踩过的坑, 逐条写进代码:
  1 `requires_grad` 上手先 grep(feedback_lr_zero_is_not_freezing_for_ddp): lr×0 不是冻结,
    DDP 仍会 all-reduce 全部梯度。这里硬断言可训参数量, 并打印。
  2 DDP 要调【包装器】不是 .module(feedback_ddp_wrapper_must_be_called_not_module):
    调 .module 等于没同步, N 卡=1 卡。
  3 梯度检查点在 eval 下静默空转(feedback_grad_checkpoint_silently_off_in_eval): 必须 train()。
  4 eval 要【全 rank】跑(feedback_eval_all_ranks), 否则有 rank 空等 ⇒ 死锁。
  5 不要按 rank 分支(feedback_ddp_deadlock_from_conditional_branches): 所有集合操作各 rank 对齐。
  6 cvm-2 多卡必须 NCCL_IB_DISABLE=1(feedback_cvm2_nccl_ib_segfault): 坏的软 RoCE 设备让
    NCCL 段错误, 单卡好、多卡全崩。由启动脚本设环境变量。
  7 单步 loss 不是趋势(feedback_per_step_loss_is_not_a_trend): 日志打滑动平均。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F


def _unpack(v):
    """把 fwd 累加出来的 9 元向量还原成可读字段。调用方只在打日志时调一次 ⇒ 一步一次 GPU 同步。
    ★这个函数必须定义在【模块级】。我第一版把它顶格插在 main() 内部的 fwd 之后 ——
      语法合法(ast.parse 过了), 但语义上把 main() 提前结束了: evaluate 和整个训练循环
      都被吞进了它的函数体, 结果一步没训、退出码还是 0。
      教训: 语法检查过不等于结构没坏, 必须断言关键代码仍在它该在的作用域里。"""
    n, na, npr, ce, cea, cep, h, ha, hp = [float(x) for x in v]
    f = lambda num, den: num / den if den > 0 else 0.0
    return {"n_sup": int(n), "n_ans": int(na), "n_prose": int(npr),
            "ce_all": f(ce, n), "ce_ans": f(cea, na), "ce_prose": f(cep, npr),
            "acc_all": f(h, n), "acc_ans": f(ha, na), "acc_prose": f(hp, npr)}


def log0(*a):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--answer-weight", type=float, default=6.0)
    ap.add_argument("--max-len", type=int, default=6656)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--save-every", type=int, default=800)
    ap.add_argument("--val-max", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--deps", default="")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--build-cache-only", action="store_true",
                    help="只建切词缓存然后退出。★必须先单独跑这一步再发 torchrun: "
                         "放在训练进程里建会让其余 rank 在屏障上等到 NCCL 超时。")
    a = ap.parse_args()

    if a.deps:
        for d in a.deps.split(","):
            if d and os.path.isdir(d) and d not in sys.path:
                sys.path.append(d)          # ★追加不前插: 前插会顶掉环境里的核心库

    if a.build_cache_only:
        os.environ.pop("RANK", None); os.environ.pop("WORLD_SIZE", None)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world > 1:
        # ★超时必须放宽: NCCL 默认 10 分钟, 而任何一个 rank 在第一次集合操作【之前】
        #   干了超过 10 分钟的活(我这里是建切词缓存 13.5 分钟), 其余 rank 就会在屏障上
        #   超时退出 —— 实测报 "setting up NCCL communicator ... store->get('0') got error"。
        #   这只是兜底; 正解是缓存改成【独立一步】先建好(--build-cache-only)。
        import datetime
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    torch.manual_seed(a.seed + rank)
    random.seed(a.seed + rank)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from answer_token_weights import WeightReport, token_weights

    tk = AutoTokenizer.from_pretrained(a.model)
    assert tk.is_fast, "★需要 fast tokenizer 才有 offset_mapping, 否则权重和 token 对不上"

    # ---------- 数据 ----------
    def build(path, cap=0):
        """cap>0 时【按轮次均匀】取, 不是取前 N 条。
        ★冒烟实测: 合并文件里 R1 在最前面, 取前 24 条 ⇒ val 里 R2/R3/R4 的 n=0,
          四轮的 val loss 只有一轮是真的, 等于没在看另外三轮。"""
        rep = WeightReport()
        lines = open(path).read().splitlines()
        if cap:
            per = {}
            for ln in lines:
                per.setdefault(json.loads(ln)["round_id"], []).append(ln)
            k = max(1, cap // max(1, len(per)))
            rnd = random.Random(a.seed)
            lines = []
            for r in sorted(per):
                pool = per[r]
                rnd.shuffle(pool)
                lines += pool[:k]
        out = []
        for ln in lines:
            d = json.loads(ln)
            m = d["messages"]
            prompt = tk.apply_chat_template([m[0], m[1]], tokenize=False,
                                            add_generation_prompt=True)
            pid = tk(prompt, add_special_tokens=False)["input_ids"]
            aid, w = token_weights(m[2]["content"], tk, a.answer_weight,
                                   report=rep, round_id=d["round_id"])
            eos = tk.eos_token_id
            aid = aid + [eos]; w = w + [1.0]
            ids = pid + aid
            if len(ids) > a.max_len:                # 实测 max_len=6656 时 0% 被截断
                continue
            lab = [-100] * len(pid) + aid
            ww = [0.0] * len(pid) + w
            out.append((ids, lab, ww, d["round_id"]))
        return out, rep

    def cache_paths(path, tag):
        st_ = os.stat(path)
        key = f"{Path(path).name}-{st_.st_size}-{int(st_.st_mtime)}-W{a.answer_weight}-L{a.max_len}-{Path(a.model).name}"
        base = Path(a.cache_dir or (Path(path).parent / "tokcache"))
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{tag}-{key}"

    def save_cache(stem, data):
        import numpy as np
        ids = np.concatenate([np.asarray(d[0], dtype=np.int32) for d in data])
        ws = np.concatenate([np.asarray(d[2], dtype=np.float16) for d in data])
        off = np.zeros(len(data) + 1, dtype=np.int64)
        for i, d in enumerate(data):
            off[i + 1] = off[i] + len(d[0])
        plen = np.asarray([sum(1 for x in d[1] if x == -100) for d in data], dtype=np.int32)
        rd = np.asarray([int(d[3][1]) for d in data], dtype=np.int8)
        np.save(f"{stem}.ids.npy", ids); np.save(f"{stem}.w.npy", ws)
        np.save(f"{stem}.off.npy", off); np.save(f"{stem}.plen.npy", plen)
        np.save(f"{stem}.rd.npy", rd)
        Path(f"{stem}.ok").write_text("ok")

    def load_cache(stem):
        import numpy as np
        ids = np.load(f"{stem}.ids.npy", mmap_mode="r")
        ws = np.load(f"{stem}.w.npy", mmap_mode="r")
        off = np.load(f"{stem}.off.npy"); plen = np.load(f"{stem}.plen.npy")
        rd = np.load(f"{stem}.rd.npy")
        out = []
        for i in range(len(plen)):
            a0, b0 = int(off[i]), int(off[i + 1])
            seq = ids[a0:b0].tolist()
            pl = int(plen[i])
            lab = [-100] * pl + seq[pl:]
            out.append((seq, lab, ws[a0:b0].astype("float32").tolist(), f"R{int(rd[i])}"))
        return out

    t0 = time.time()
    tr_stem, va_stem = cache_paths(a.train, "train"), cache_paths(a.val, "val")
    # ★只让 local_rank 0 建缓存, 其余等它 —— 16 个 rank 各切一遍要 18 分钟, 还各占 19GB 内存
    need = not (Path(f"{tr_stem}.ok").exists() and Path(f"{va_stem}.ok").exists())
    if a.build_cache_only:
        if not need:
            print(f"  缓存已存在, 无需重建: {tr_stem}", flush=True)
            return
        tr_d, rep = build(a.train)
        va_d, _ = build(a.val, cap=a.val_max)
        save_cache(tr_stem, tr_d); save_cache(va_stem, va_d)
        print(f"  缓存已建 ({time.time()-t0:.0f}s) train {len(tr_d):,} val {len(va_d):,}", flush=True)
        print(f"  答案 token 占 assistant {rep.answer_tokens/max(rep.tokens,1):.2%} | "
              f"跨界 token {rep.straddling_tokens}", flush=True)
        assert rep.straddling_tokens == 0, "★有跨界 token"
        return
    assert not need, ("★缓存不存在。先在【每台机器】上单独跑一次 "
                      "`--build-cache-only`, 再发 torchrun —— "
                      "在训练进程里建缓存会让其余 rank 在屏障上等到 NCCL 超时。")
    if False:
        pass
    if True:
        train, val = load_cache(tr_stem), load_cache(va_stem)
        rep = WeightReport()
        # ★缓存必须硬校验: 数据路径静默损坏比省 18 分钟可怕。随机抽 8 条现场重切并逐一比对。
        probe, _ = build(a.val, cap=a.val_max)
        chk = {tuple(x[0]): x for x in probe}
        import random as _r
        hit = 0
        for v_ in _r.Random(0).sample(val, min(8, len(val))):
            k_ = tuple(v_[0])
            assert k_ in chk, "★缓存里的 token 序列和现场重切不一致"
            assert v_[1] == chk[k_][1], "★缓存的 labels 和现场重切不一致"
            assert max(abs(x - y) for x, y in zip(v_[2], chk[k_][2])) < 1e-3, "★缓存的权重不一致"
            hit += 1
        log0(f"  缓存已载并校验 {hit}/8 条逐一对上 ({time.time()-t0:.0f}s)")
        rep.tokens = 1; rep.answer_tokens = 0; rep.weight_sum = 1; rep.answer_weight_sum = 0
    log0(f"  train {len(train):,} | val {len(val):,} | 读+切词 {time.time()-t0:.0f}s")
    log0(f"  答案 token 占 assistant {rep.answer_tokens/max(rep.tokens,1):.2%} | "
         f"跨界 token {rep.straddling_tokens} | "
         f"W={a.answer_weight} 时答案梯度份额 {rep.answer_weight_sum/max(rep.weight_sum,1):.1%}")
    if rep.samples:
        assert rep.straddling_tokens == 0, "★有跨界 token, 权重会落到非答案字符上"

    # ★按长度分桶再成批: R1 的 token 数只有 R4 的约 1/4, 朴素成批会浪费约 40% 的算力在 padding 上
    def batches(data, mb, shuffle, seed, group=1):
        idx = list(range(len(data)))
        if shuffle:
            rnd = random.Random(seed)
            rnd.shuffle(idx)
            buf = mb * 64
            chunks = [idx[i:i + buf] for i in range(0, len(idx), buf)]
            idx = []
            for c in chunks:
                c.sort(key=lambda j: len(data[j][0]))
                idx += c
        else:
            idx.sort(key=lambda j: len(data[j][0]))
        bs = [idx[i:i + mb] for i in range(0, len(idx), mb)]
        if shuffle:
            # ★必须按 group(=world) 个一组打乱, 不能逐批打乱。
            #   `bs[rank::world]` 把同一步的 world 个批分给各 rank, 而每步耗时由【最慢那张卡】
            #   决定。逐批打乱会让同一步里各卡的序列长度差很多 ——
            #   实测: 单卡同配置 8.7s/步(批内长度波动 6.5-27.9s), 16 卡变 15.3s/步, 1.76 倍的拖尾。
            #   按组打乱则组内 world 个批在长度排序上相邻 ⇒ 各卡长度相近。
            g = [bs[i:i + group] for i in range(0, len(bs), group)]
            random.Random(seed + 1).shuffle(g)
            bs = [b for gg in g for b in gg]
        return bs

    def collate(data, sel):
        n = max(len(data[j][0]) for j in sel)
        pad = tk.pad_token_id if tk.pad_token_id is not None else tk.eos_token_id
        I = torch.full((len(sel), n), pad, dtype=torch.long)
        L = torch.full((len(sel), n), -100, dtype=torch.long)
        W = torch.zeros(len(sel), n, dtype=torch.float)
        A = torch.zeros(len(sel), n, dtype=torch.long)
        for b, j in enumerate(sel):
            ids, lab, ww, _ = data[j]
            I[b, :len(ids)] = torch.tensor(ids); L[b, :len(lab)] = torch.tensor(lab)
            W[b, :len(ww)] = torch.tensor(ww); A[b, :len(ids)] = 1
        return I, L, W, A

    # ---------- 模型 ----------
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa")
    except TypeError:          # 不同 transformers 版本参数名不一样, 不猜, 退回默认
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    model.config.use_cache = False
    lc = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                    bias="none", task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lc)
    # ★use_reentrant=False: 重入版的 checkpointing 会让 DDP 的梯度 all-reduce 无法和 backward
    #   重叠(autograd 钩子挂在重算图上), 这是 checkpointing + DDP 的已知冲突。
    #   我的单卡基准没有 DDP 包装, 所以这一项在单卡上根本测不出来 —— 这就是单卡 8.7s
    #   外推到 16 卡会错的原因。
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = model.cuda()
    model.train()                       # ★检查点在 eval 下静默空转, 必须 train()

    # ★坑1: 上手先 grep requires_grad
    tr = [(n_, p.numel()) for n_, p in model.named_parameters() if p.requires_grad]
    ntr = sum(x[1] for x in tr)
    ntot = sum(p.numel() for p in model.parameters())
    log0(f"  可训参数 {ntr/1e6:.1f}M / 总 {ntot/1e9:.2f}B = {ntr/ntot:.3%} | 张量 {len(tr)} 个")
    log0(f"    例: {[x[0] for x in tr[:2]]}")
    assert ntr > 0, "★没有可训参数"
    assert ntr / ntot < 0.05, f"★可训占比 {ntr/ntot:.1%} 太高, LoRA 没挂上?"

    if world > 1:
        # ★static_graph=True: checkpointing 下官方要求, 同时让 DDP 能预先规划分桶。
        # ★bucket_cap_mb 调大: LoRA 梯度只有 116MB 但分成 256 个张量, 默认 25MB 的桶
        #   会切成 5 个桶 ⇒ 5 次跨机 all-reduce 各吃一次延迟。调到 256MB 合成 1 个。
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=False,
            gradient_as_bucket_view=True, static_graph=True, bucket_cap_mb=256)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    bs_all = batches(train, a.micro_batch, True, a.seed, group=world)
    per_rank = len(bs_all) // world
    steps_per_epoch = max(1, per_rank // a.accum)
    total_steps = int(steps_per_epoch * a.epochs)
    log0(f"  micro-batch {a.micro_batch} x accum {a.accum} x world {world} "
         f"= 全局 {a.micro_batch*a.accum*world} 条/步 | 每轮 {steps_per_epoch} 步 | "
         f"总 {total_steps} 步")

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, total_steps - a.warmup)
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    def fwd(I, L, W, A):
        """返回 (加权loss, 细分统计 dict)。

        ★为什么必须把【答案 token】和【散文 token】的损失分开报
          (feedback_mixed_bucket_metric_hides_collapse): 答案只占 assistant token 的 4.7%,
          总体 loss 几乎全是散文。散文在降而答案不降, 看总体是看不出来的 ——
          而我们唯一在意的就是答案。另外报答案 token 的 top-1 准确率:
          那是个能直接读的数(teacher forcing 下预测对了几个位置/氨基酸 token)。
        """
        # ★绝不能对整条序列的 logits 调 .float(): Qwen3.5-9B 的 vocab 是 248,077,
        #   batch 2 x 6656 x 248077 x 4B = 13.2GB 的 float32, 直接把卡吃满。
        #   先按【被监督的位置】选出来(每条约 772 个)再转 float32 ⇒ 约 1.5GB。
        out = model(input_ids=I.cuda(), attention_mask=A.cuda())   # ★调包装器不是 .module
        lg = out.logits[:, :-1]                                    # 保持 bf16
        lab = L[:, 1:].cuda(); w = W[:, 1:].cuda()
        m = lab.ne(-100)
        # 构造上每条样本都有被监督的 token(assistant 段非空), 所以这里断言而不是走空分支 ——
        # 空分支里 `logits.sum()*0.0` 遇到 -inf 会变 NaN(feedback_neg_inf_times_zero_is_nan)。
        assert bool(m.any()), "★这一批没有任何被监督的 token, 数据构造有问题"
        sel = lg[m].float()
        gold = lab[m]
        ce = F.cross_entropy(sel, gold, reduction="none")
        ww = w[m]
        loss = (ce * ww).sum() / ww.sum().clamp(min=1e-6)
        with torch.no_grad():
            # ★统计必须【留在 GPU 上】累加, 不能每个 micro-batch 调 float()。
            #   每次 float() 都是一次 GPU->CPU 同步, 原来 7 个字段 x 2 个 micro-batch
            #   = 每步 14 次同步, 把计算和梯度通信的重叠全打断了。
            #   这里返回一个 9 元向量, 只在打日志时同步一次。
            hit = sel.argmax(-1).eq(gold).float()
            is_ans = ww.gt(1.0)                 # 权重 >1 的就是答案 token(散文恒为 1.0)
            pr = ~is_ans
            v = torch.stack([
                m.sum().float(), is_ans.sum().float(), pr.sum().float(),
                ce.sum(), (ce * is_ans).sum(), (ce * pr).sum(),
                hit.sum(), (hit * is_ans).sum(), (hit * pr).sum(),
            ])
        return loss, v

    @torch.no_grad()
    def evaluate():
        """按轮报【答案 token】和【散文 token】各自的 CE 与 top-1 准确率。
        ★全 rank 都跑各自一份再 all_reduce(feedback_eval_all_ranks): 只让 rank0 跑会让
          其他 rank 空等在后面的集合操作上 ⇒ 死锁。"""
        model.eval()
        keys = ["R1", "R2", "R3", "R4"]
        FIELDS = 6          # ce_ans, hit_ans, n_ans, ce_prose, hit_prose, n_prose
        byr = {k: [0.0] * FIELDS for k in keys}
        vb = batches(val, a.micro_batch, False, 0)
        for sel in vb[rank::world]:
            I, L, W, A = collate(val, sel)
            _, v = fwd(I, L, W, A)
            b = byr[val[sel[0]][3]]
            b[0] += float(v[4]); b[1] += float(v[7]); b[2] += float(v[1])
            b[3] += float(v[5]); b[4] += float(v[8]); b[5] += float(v[2])
        model.train()                   # ★梯度检查点在 eval 下空转, 必须切回来
        v = torch.zeros(len(keys) * FIELDS, device="cuda")
        for i, k in enumerate(keys):
            for j in range(FIELDS):
                v[i * FIELDS + j] = byr[k][j]
        if world > 1:
            dist.all_reduce(v)
        out = {}
        for i, k in enumerate(keys):
            b = [float(x) for x in v[i*FIELDS:(i+1)*FIELDS]]
            out[k] = {"ce_ans": b[0]/max(b[2],1), "acc_ans": b[1]/max(b[2],1),
                      "n_ans": int(b[2]),
                      "ce_prose": b[3]/max(b[5],1), "acc_prose": b[4]/max(b[5],1)}
        return out

    Path(a.out).mkdir(parents=True, exist_ok=True)
    vsum = torch.zeros(9, device="cuda"); lsum = torch.zeros(1, device="cuda")
    gsum = torch.zeros((), device="cuda"); nacc = 0; tok_sum = 0
    rvec = {}
    step = 0
    tic = time.time()
    done = False
    ep = 0
    while not done:
        bs_all = batches(train, a.micro_batch, True, a.seed + ep, group=world)
        mine = bs_all[rank::world][:per_rank]
        for i in range(0, len(mine) - a.accum + 1, a.accum):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            vacc = torch.zeros(9, device="cuda")
            lacc = torch.zeros(1, device="cuda")
            tall = 0
            for j in range(a.accum):
                sel = mine[i + j]
                I, L, W, A = collate(train, sel)
                loss, v = fwd(I, L, W, A)
                (loss / a.accum).backward()
                vacc += v                      # ★全程在 GPU 上累加, 不同步
                lacc += loss.detach()
                tall += int(A.sum())           # 这是 CPU 侧张量, 不触发 GPU 同步
                for r in {train[t][3] for t in sel}:
                    rvec.setdefault(r, torch.zeros(3, device="cuda"))
                    rvec[r] += torch.stack([v[4], v[7], v[1]])   # ce_ans和, hit_ans和, n_ans
            gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                                    if p.requires_grad], 1.0)
            opt.step()
            step += 1
            # ★只在要打日志的那一步同步, 其余步数一次 GPU->CPU 传输都不做
            vsum += vacc
            lsum += lacc
            gsum += gnorm.detach()
            nacc += 1
            tok_sum += tall
            if step % a.log_every == 0:
                st_ = _unpack(vsum)                       # 这里才同步, 一步一次
                mv = float(lsum) / max(nacc * a.accum, 1)
                gm = float(gsum) / max(nacc, 1)
                dt = max(time.time() - tic, 1e-6)
                log0(f"  步 {step}/{total_steps} | 加权 {mv:.4f} | "
                     f"★答案 CE {st_['ce_ans']:.4f} 准确 {st_['acc_ans']:.1%} | "
                     f"散文 CE {st_['ce_prose']:.4f} 准确 {st_['acc_prose']:.1%} | "
                     f"全体 CE {st_['ce_all']:.4f} 准确 {st_['acc_all']:.1%}")
                # ★tok/s 用【这 nacc 步的累计 token】, 不是用最后一步代表全部 ——
                #   原来那个写法在长度分桶下误差 4 倍, 害我据此推出一个错的"13s 固定开销"。
                log0(f"      grad {gm:.2f} | lr {lr_at(step):.2e} | "
                     f"总 {tok_sum*world/dt/1000:.1f}k tok/s "
                     f"(监督 {st_['n_sup']*world/dt/1000:.1f}k) | "
                     f"{dt/nacc:.1f}s/步 | 预计剩 {(total_steps-step)*dt/nacc/3600:.1f}h")
                if rvec:
                    parts = []
                    for r in sorted(rvec):
                        c, h, n_ = [float(x) for x in rvec[r]]
                        if n_:
                            parts.append(f"{r} CE {c/n_:.3f}/准确 {h/n_:.0%}")
                    log0("      按轮(答案 token): " + "  ".join(parts))
                    rvec.clear()
                vsum.zero_(); lsum.zero_(); gsum.zero_(); nacc = 0; tok_sum = 0
                tic = time.time()
            if step % a.eval_every == 0 or step == total_steps:
                e = evaluate()
                log0("  [val] ★答案 token: " + "  ".join(
                    f"{k} CE {e[k]['ce_ans']:.4f}/准确 {e[k]['acc_ans']:.1%}(n={e[k]['n_ans']})"
                    for k in e))
                log0("  [val]   散文 token: " + "  ".join(
                    f"{k} CE {e[k]['ce_prose']:.4f}/准确 {e[k]['acc_prose']:.1%}" for k in e))
            if step % a.save_every == 0 or step == total_steps:
                if rank == 0:
                    tgt = model.module if world > 1 else model
                    tgt.save_pretrained(f"{a.out}/step{step}")
                    log0(f"  已存 {a.out}/step{step}")
                if world > 1:
                    dist.barrier()
            if step >= total_steps:
                done = True
                break
        ep += 1
    log0("  训练完成")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
