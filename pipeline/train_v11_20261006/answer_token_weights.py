# -*- coding: utf-8 -*-
"""给 assistant 段的每个 token 算损失权重: 散文 1.0, 真正的答案(位置+氨基酸) W。

★为什么需要(实测, 抽样 400 条 sft_v10_merged.jsonl 的平均字符数):
    提示词(system+user)  5966   ← assistant 的 1.9 倍, 必须 -100 掩掉
    assistant 总         3098
      reasoning          1143   36.9%
      why(逐位置理由)       963   31.1%
      uncertainty         238    7.7%
      ★答案(位置+氨基酸)      63    2.0%   ← 我们唯一衡量的东西
  ★但【字符占比不等于 token 占比】: 用 Qwen3.5-9B 的 tokenizer 在 2,000 条真样本上实测,
    答案占 token 的 **6.54%**(不是 2.0%)。因为 BPE 把位置号 166 切成 '1','6','6' 三个 token
    (一位数字一个), 而散文一个 token 装好几个字符 ⇒ 答案在 token 维度比字符维度"贵" 3 倍。
    选 W 必须按 token 占比反解, 按字符占比会把 W 定大一倍:
        W=1 → 6.5%   W=4 → 21.9%   W=6 → ~30%   W=8 → 35.9%   W=12 → 45.6%   W=20 → 58.3%

  等权监督 ⇒ 93.5% 的梯度在散文上。但【不能只监督答案】:
    - 推理时模型要先生成 reasoning 再出 picks; reasoning 没被监督就会吐垃圾, 答案条件在垃圾上
    - 我们自己两次把东西从损失里拿掉都毁了能力:
      copy_weight_zero_destroys_the_read_path (97.15%→17.71%, CE 8.954=频率表的熵)
      stage2_rounds_regressed_to_the_prior (送分题不进损失→毁掉读输入通路)
  ⇒ 结论: assistant 整段都监督, 但答案 token 加权。
    建议测 W∈{1, 6, 12} (对应答案梯度 6.5% / 30% / 46%), 别测 1 vs 2 —— 差异太小量不出来
    (feedback_underpowered_config_change)。

★两个实现上的坑, 都在这里处理掉了:
  1 不能用字符串查找定位答案。位置号 "11" 可能正好出现在某个 why 文本里,
    str.find 会标错地方。这里改成【按 assembler 的写法逐段重建 JSON 并记录偏移】,
    重建完硬断言和存盘的字符串逐字节相同, 不同就报错退出, 绝不带着错偏移继续。
  2 BPE 会把 :"D" 之类合成一个 token。只要 token 的字符区间和答案区间【有重叠】就算答案,
    宁可多包一点格式字符, 也不能漏掉答案本身; 重叠量会统计出来让代价可见。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class WeightReport:
    """一批样本上的实测统计, 用来确认权重真的落在该落的地方。"""
    samples: int = 0
    tokens: int = 0
    answer_tokens: int = 0
    weight_sum: float = 0.0
    answer_weight_sum: float = 0.0
    straddling_tokens: int = 0          # 同时覆盖答案和散文的 token
    per_round: dict = field(default_factory=dict)

    @property
    def answer_token_share(self) -> float:
        return self.answer_tokens / self.tokens if self.tokens else 0.0

    @property
    def answer_gradient_share(self) -> float:
        """答案在损失里实际拿到的份额 = 答案权重和 / 总权重和。"""
        return self.answer_weight_sum / self.weight_sum if self.weight_sum else 0.0


def answer_spans(assistant_text: str) -> list[tuple[int, int]]:
    """返回答案(位置号与氨基酸字母)在 assistant 字符串里的字符区间 [起, 止)。

    按 assemble_sft_v10.py 的写法逐段重建:
        json.dumps({"reasoning":..., "picks":[...], "uncertainty":...},
                   separators=(",", ":"), ensure_ascii=False)
    重建结果必须和入参逐字节相同, 否则抛错(宁可失败也不能标错位置)。
    """
    obj = json.loads(assistant_text)
    if "picks" not in obj:
        raise ValueError("assistant 段没有 picks 字段, 这不是 v10 的格式")

    def dump(value) -> str:
        return json.dumps(value, ensure_ascii=False)

    buf: list[str] = []
    pos = 0
    spans: list[tuple[int, int]] = []

    def emit(text: str) -> None:
        nonlocal pos
        buf.append(text)
        pos += len(text)

    def emit_tracked(text: str) -> None:
        """写入并把这段记成答案。"""
        nonlocal pos
        start = pos
        emit(text)
        spans.append((start, pos))

    emit('{"reasoning":')
    emit(dump(obj.get("reasoning", "")))
    emit(',"picks":[')
    for i, p in enumerate(obj["picks"]):
        if i:
            emit(",")
        emit('{"position_1based":')
        emit_tracked(dump(p["position_1based"]))      # 答案: 位置号(数字, 没有引号)
        emit(',"aa":')
        # ★氨基酸要【只跟踪字母本身, 不含引号】。
        #   dump("Q") == '"Q"', 带引号的话前后那两个纯格式 token('":"' 和 '","')
        #   会因为含引号被判成答案 —— 实测这样跨界 token 占到答案 token 的 35.3%。
        aa_json = dump(p["aa"])
        if len(aa_json) >= 2 and aa_json[0] == '"' and aa_json[-1] == '"':
            emit('"')                                # 开引号: 格式
            emit_tracked(aa_json[1:-1])              # 答案: 氨基酸字母
            emit('"')                                # 闭引号: 格式
        else:                                        # 非字符串(不该发生)就整段跟踪
            emit_tracked(aa_json)
        emit(',"why":')
        emit(dump(p.get("why", "")))
        emit("}")
    emit('],"uncertainty":')
    emit(dump(obj.get("uncertainty", "")))
    emit("}")

    rebuilt = "".join(buf)
    if rebuilt != assistant_text:
        raise ValueError(
            "重建的 assistant JSON 与原文不一致 ⇒ 偏移不可信, 拒绝继续。"
            f" 原文 {len(assistant_text)} 字, 重建 {len(rebuilt)} 字")
    return spans


def token_weights(assistant_text: str, tokenizer, answer_weight: float,
                  *, prose_weight: float = 1.0, report: WeightReport | None = None,
                  round_id: str | None = None) -> tuple[list[int], list[float]]:
    """把 assistant 段切成 token 并给出每个 token 的权重。

    返回 (input_ids, weights)。调用方必须用同一个 tokenizer、同样的
    add_special_tokens=False 去切 assistant 段, 否则权重和 token 对不上。
    """
    enc = tokenizer(assistant_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    if len(ids) != len(offsets):
        raise ValueError("tokenizer 没有给出等长的 offset_mapping; 需要 fast tokenizer")

    spans = answer_spans(assistant_text)
    weights: list[float] = []
    straddle = 0
    n_ans = 0
    for (a, b) in offsets:
        if b <= a:                      # 特殊 token 没有字符区间
            weights.append(prose_weight)
            continue
        hit = any(a < e and s < b for (s, e) in spans)     # 区间有重叠
        if hit:
            n_ans += 1
            weights.append(answer_weight)
            inside = sum(min(b, e) - max(a, s) for (s, e) in spans if a < e and s < b)
            if inside < (b - a):        # 这个 token 越界包进了非答案字符
                straddle += 1
        else:
            weights.append(prose_weight)

    if report is not None:
        report.samples += 1
        report.tokens += len(ids)
        report.answer_tokens += n_ans
        report.weight_sum += sum(weights)
        report.answer_weight_sum += n_ans * answer_weight
        report.straddling_tokens += straddle
        if round_id:
            r = report.per_round.setdefault(round_id, {"tokens": 0, "answer": 0})
            r["tokens"] += len(ids)
            r["answer"] += n_ans
    return ids, weights
