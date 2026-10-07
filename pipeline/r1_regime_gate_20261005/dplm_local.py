# -*- coding: utf-8 -*-
"""本地 H100 上加载 DPLM-bit 的最小封装。生产机一张卡都不碰。

★不装、不改任何包(feedback_dep_install_downgrades_core_lib: 装依赖会偷偷降核心库)。
  byprot 的 import 链只缺一个符号 —— lightning 2.6 把
  `_has_meta_device_parameters` 改名成 `_has_meta_device_parameters_or_buffers`,
  这里只做一个别名, 不落盘。
★六个接口坑逐条照抄 gen_round_drafts.py 的记录(不要凭记忆改):
  1 token id 一律用模型属性(tokenizer 查出来的 struct mask 是 8229, 模型属性是 8228, 差一位全废)
  2 generate 返回 dict, 键 output_tokens / res_mask / final_struct_feature
  3 partial_masks=True 表示"给定/冻结"
  4 两半必须等长: [struct_bos + L + struct_eos] ++ [aa_bos + L + aa_eos] (chunk(2,dim=1) 硬编码)
  5 struct_tokenizer 用 strict=False 加载 ⇒ 必须断言 "0 missing and 0 unexpected"
  6 vocab 与 struct_tokenizer 必须显式指本地
"""
import contextlib
import io
import os
import sys

P = "/lambda/nfs/file2/cqr_files/protein_design"
MODEL = f"{P}/models/dplm2_bit_650m"
STOK = f"{P}/models/dplm2_struct_tokenizer_hydra_wrapper"
# ★用从 cvm-2 抄回来的那份源码, 不用本地 alt_backbones 那份:
#   本地那份 `from transformers.models.esm.modeling_esm import *` 在 transformers 5.x 下
#   因为上游加了 __all__ 而拿不到它要继承的内部类(EsmSelfAttention 未定义);
#   cvm 那份已经把 4.39 的 ESM 冻结进 `byprot.models.dplm2.modules._esm439`。
DPLM_SRC = f"{P}/code/dplm_cvm"


# ★堵掉而不是空壳: plm 环境里的 torchvision 与 torch 2.2.1 不匹配(循环 import 抛 AttributeError),
#   而 transformers 的 image_utils 用 is_vision_available() 做分支 —— 它只认 ImportError。
#   `sys.modules[name] = None` 会让 `import name` 干净地抛 ImportError ⇒ transformers 走无视觉分支。
#   DPLM-bit 的推理路径不碰视觉。
_BLOCK = ("torchvision", "torchaudio")


def _block_modules():
    for n in _BLOCK:
        if n not in sys.modules:
            sys.modules[n] = None


def _shim_stdlib():
    """Python 3.12 删了 `imp`, 而 byprot 里有一处 `import imp` 是【没用到的】
    (grep 不到任何 `imp.` 调用) ⇒ 塞个空壳即可, 不用换 python、不用装包。"""
    import types
    if "imp" not in sys.modules:
        try:
            import imp  # noqa: F401
        except ModuleNotFoundError:
            sys.modules["imp"] = types.ModuleType("imp")


def _shim_lightning():
    import lightning_fabric.strategies.fsdp as m
    if not hasattr(m, "_has_meta_device_parameters"):
        from lightning_fabric.utilities.init import _has_meta_device_parameters_or_buffers as f
        m._has_meta_device_parameters = f


# 纯数据加载用的重包: 推理(generate / detokenize)完全不碰。装它们要编译, 还会牵动 numpy。
# ★空壳掉的模块【全部打印出来】便于审计: 真被用到会在调用处抛 AttributeError 而不是静默,
#   而生成结果另有一道几何硬闸(相邻 CA 距离中位必须 3.6-4.0 Å)兜底。
_STUBBED = []


def _import_with_stubs(max_rounds=25):
    import types
    for _ in range(max_rounds):
        try:
            from byprot.models.dplm2.dplm2_bit import DPLM2Bit
            if _STUBBED:
                print(f"  已空壳(仅数据加载用, 推理不碰): {', '.join(_STUBBED)}", flush=True)
            return DPLM2Bit
        except ImportError as e:
            name = getattr(e, "name", None)
            if not name or name.startswith("byprot") or name in _STUBBED:
                raise
            mod = types.ModuleType(name)
            mod.__path__ = []

            def _mk(_n):
                def _getattr(k):
                    # ★取属性时不能炸: `from torch_scatter import scatter` 会先取属性,
                    #   取不到 Python 会转成 ImportError, 那样循环就卡住了。
                    #   返回一个【被调用才炸】的占位 ⇒ 真用到才暴露, 且暴露得很响。
                    def _boom(*a, **k2):
                        raise RuntimeError(
                            f"★{_n}.{k} 被空壳掉了却真被调用 ⇒ 推理路径用得到它, 必须真装")
                    _boom.__name__ = k
                    return _boom
                return _getattr

            mod.__getattr__ = _mk(name)
            sys.modules[name] = mod
            _STUBBED.append(name)
            for m in [k for k in sys.modules if k.startswith("byprot")]:
                del sys.modules[m]
    raise SystemExit(f"空壳 {max_rounds} 轮还没导入成功, 已空壳: {_STUBBED}")


def load():
    for p in (DPLM_SRC, f"{DPLM_SRC}/src", f"{DPLM_SRC}/vendor/openfold"):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    # ★缺的包装在本目录 pydeps/, 【追加】到 sys.path 末尾而不是插到前面 ——
    #   前插会让 pydeps 里的 absl 顶掉 base 里的, 那正是"装依赖偷偷换掉核心库"那类事故。
    # py3.10 用 pydeps310(老版 lightning 自带 _has_meta_device_parameters);
    # py3.13 用 pydeps。★两套分开放, 不要混 —— 混了就是"同名 venv 差 8 个包"那类事故。
    here = os.path.dirname(os.path.abspath(__file__))
    dep = os.path.join(here, "pydeps310" if sys.version_info[:2] == (3, 10) else "pydeps")
    assert os.path.isdir(dep), f"缺依赖目录 {dep}"
    if dep not in sys.path:
        sys.path.append(dep)
    # struct_tokenizer 的 GVP 编码器在构造时会去 torch hub 拉 ESM-IF 权重(约 1.5GB),
    # 固定缓存目录, 免得每次重下。★这条在真正的推理路径上, 不能空壳掉。
    os.environ.setdefault("TORCH_HOME", f"{P}/torch_hub")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_CACHE", f"{P}/hf_cache_dplm2/hub")
    _block_modules()
    _shim_stdlib()
    _shim_lightning()
    DPLM2Bit = _import_with_stubs()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m = DPLM2Bit.from_pretrained(MODEL, cfg_override={"tokenizer": {"vocab_file": MODEL},
                                                          "struct_tokenizer": {"exp_path": STOK}})
    log = buf.getvalue()
    assert "0 missing and 0 unexpected" in log, f"★struct_tokenizer 加载不干净:\n{log[-500:]}"
    print("  struct_tokenizer: 0 missing and 0 unexpected ✅", flush=True)
    return m.eval().cuda()
