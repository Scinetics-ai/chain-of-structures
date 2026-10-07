双 CVM SGLang teacher 部署记录（2026-10-03）

两台 CVM 各有 8 张 NVIDIA RTX PRO 5000 **72GB Blackwell**（SM120），按每台 4 个独立 TP2 副本部署，共 8 个服务、16 张卡。每个副本仅在本机两张卡内做 tensor parallel；两机无 IB，跨机采用独立数据任务并行分发，不组成跨机 TP。服务绑定 `127.0.0.1`，OpenAI API 模型名为 `Qwen/Qwen3.8-27B`。

当前工作仅完成部署、容量与有限吞吐诊断。`full_collection_allowed=false`、`student_training_allowed=false`，full SFT hold 保持；这些结果未解除 teacher 语义质量门。

| 项目 | CVM1 | CVM2 |
| --- | --- | --- |
| SSH 别名 | `cvm-1` | `cvm-2` |
| SSH 主机 | `ubuntu@42.192.205.139` | `ubuntu@124.221.97.180` |
| 工作根目录 | `/data/caiqingran/sglang_teacher_20261003` | 同左 |
| Python | `/data/caiqingran/sglang_teacher_20261003/runtime/bin/python` | 同左 |
| 模型目录 | `/data/caiqingran/models/models/Qwen3.8-27B` | `/data/caiqingran/models/Qwen3.8-27B` |
| 启动、停止脚本 | 工作根目录下 `launch_replica.py`、`stop_replica.py` | 同左 |

两台 CVM 使用从已验证 A800 环境复制的同一套 runtime：Python 3.12.15、SGLang `0.5.19+cu129audit1`、PyTorch `2.13.0+cu129`、sglang-kernel `0.4.6.post1+cu129`、FlashInfer `0.6.18`、Transformers `5.12.1`、XGrammar `0.2.1`。CVM 的 CUDA 13 系统工具链与该 cu129 runtime 已实际完成 FlashInfer JIT、CUDA graph、health 和短请求验证。运行环境及 GPU 证据见 [CVM2 runtime 验证](cvm2/runtime_verification.json)。

A800、CVM1、CVM2 的 18 个权重分片逐文件 SHA256 一致，总计 55,563,006,776 字节；模型 config 和权重 index 也已核对。路径名称不同不代表模型不同。证据见 [模型一致性记录](model_identity_comparison.json) 和 [CVM2 对 A800 的比对](cvm2/model_identity_vs_a800.json)。

硬件宿主环境不完全相同：A800 驱动为 550.90.12、默认功耗上限 300W；CVM 驱动为 580.126.20、PRO 5000 功耗上限 350W。系统 CUDA 编译工具链也不同。因此相同权重、Python runtime 和请求的结果比较的是这两套实际部署，不是排除驱动、拓扑和功耗影响后的纯芯片基准。

每台主机的最终端口与 GPU 分配如下。每个实例的当前 PID、进程启动 ticks 和完整命令以对应 `runs/<run>/launch.json` 为准，文档不固定 PID。

| 本机端口 | 本机 GPU | CVM1 run | CVM2 run |
| --- | --- | --- | --- |
| 31038 | 0,1 | `tp2_g01_p31038_v2` | `tp2_g01_p31038_v2` |
| 31039 | 2,3 | `tp2_g23_p31039_v2` | `tp2_g23_p31039_v3` |
| 31042 | 4,5 | `tp2_g45_p31042_v2` | `tp2_g45_p31042_v1` |
| 31043 | 6,7 | `tp2_g67_p31043_v2` | `tp2_g67_p31043_v1` |

统一 TP2 参数：`--dtype bfloat16 --tp-size 2 --context-length 32768 --mem-fraction-static 0.85 --chunked-prefill-size 4096 --max-running-requests 16 --attention-backend flashinfer --reasoning-parser qwen3`，不量化、不启用 speculative decoding。启动环境使用 `NCCL_IB_DISABLE=1`、`NCCL_SOCKET_IFNAME=eth0`、`OMP_NUM_THREADS=4`、`PYTHONNOUSERSITE=1`；PATH 包含 `runtime/bin` 和 `/usr/local/cuda/bin`，保证 ninja/JIT 工具可见。启动器清理继承的 `PYTHONPATH`、`PYTHONHOME`，并将编译缓存放在本任务 `runs/cache` 下。

部署中的 `max-running-requests=16` 是每副本服务上限；测量的 c4/c8/c16 是客户端并发。SGLang 自动 NUMA 绑定默认会尝试启用；server_info 中 `numa_node=null` 仅表示没有显式指定节点，不等于已成功绑定。现有回执不能据此认定 CPU 或 NUMA 是瓶颈，详见 [吞吐审计](pool_fairness_audit.json)。

已完成的有限吞吐诊断如下，均为单副本结果。每阶段使用同一批 16 个真实四轮标注 payload，共 171,594 个 prompt tokens；每次运行 warmup 4 请求，各测量阶段前 flush。固定生成参数为 `max_tokens=2048, temperature=0.2, top_p=0.9, seed=20261003, enable_thinking=false`，non-streaming，不强制忽略 EOS、不自动重试。

| 主机与配置 | 并发 | 实际输出 tokens | 客户端阶段秒数 | 输出 tok/s | stop / length | API 错误 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A800 TP2 | 4 | 30,346 | 213.220 | 142.32 | 12 / 4 | 0 |
| A800 TP2 | 8 | 31,025 | 132.173 | 234.73 | 11 / 5 | 0 |
| CVM1 PRO 5000 TP2 | 4 | 30,519 | 280.847 | 108.67 | 10 / 6 | 0 |
| CVM1 PRO 5000 TP2 | 8 | 30,738 | 164.892 | 186.41 | 11 / 5 | 0 |
| CVM1 PRO 5000 TP2 | 16 | 30,489 | 151.680 | 201.01 | 11 / 5 | 0 |
| CVM2 PRO 5000 TP2 | 4 | 30,465 | 294.098 | 103.59 | 10 / 6 | 0 |
| CVM2 PRO 5000 TP2 | 8 | 30,658 | 168.587 | 181.85 | 11 / 5 | 0 |
| CVM2 PRO 5000 TP1 | 8 | 30,511 | 385.927 | 79.06 | 11 / 5 | 0 |
| A800 TP1 | 8 | 30,363 | 263.515 | 115.22 | 11 / 5 | 0 |

数据来源：[A800 TP2](a800_tp2_real_01/summary.json)、[CVM1 TP2](cvm1_tp2_c4_c8_c16/summary.json)、[CVM2 TP2](cvm2_tp2_real_01/summary.json)、[CVM2 TP1](cvm2_tp1_g3_c8_real_01/summary.json)。A800/CVM1/CVM2 TP2 使用相同 frozen harness SHA；TP1 使用增补并发选项的 runner，其相同请求内容与 prompt tokens 已逐条核对，见 [CVM2 配置比对](cvm2_profile_comparison_01.json)。

[A800 TP1 的实测](a800_tp1_tuning/a800_tp1_c8_real_01/summary.json)与 CVM2 TP1 使用同一 runner、BF16、并发 8 和显存/缓存配置；该组 A800 输出吞吐高 45.7%。相同 TP2、并发 8 时 A800 比 CVM1 高 25.9%。这些是本批请求在两套主机上的差距；TP1 的 CVM2 测量还与另一 GPU 组上的 TP2 测量并行，共享主机 CPU。

A800 的临时单卡测速服务已停止；GPU2,3 的原双卡 31039 服务已恢复并返回 health=200，GPU0,1 原服务进程未被改变。

CVM2 TP1 容量诊断采用 GPU3、`.90` 显存比例、chunk 2048、max-running 8、max-mamba-cache-size 40、max-total-tokens 98304；该结果不能视为与 TP2 完全同配置。它与同机 GPU0,1 上的 TP2 测量同时运行，包含共享 CPU 等主机资源的影响。此次 TP1 c8 的 79.06 tok/s/卡低于同机 TP2 c8 的 90.93 tok/s/卡；CVM1 单副本 TP2 c16 为 100.50 tok/s/卡。因此本次继续采用四 TP2/主机、池测量 c16，并不宣称所有配置中的全局最优，也不把单副本结果乘以 8 当作已测 16 卡吞吐。

SGLang 0.5.19 的 `/flush_cache` 同时清理 prefix/KV 和 XGrammar 编译缓存。因此 warmup 后的各测量阶段仍是 **cold grammar + cold prefix cache 的有限批次**，不是稳定预热状态下的峰值。对应的官方源码调用链和行号保存在 [审计记录](pool_fairness_audit.json)：`scheduler.flush_cache → grammar_manager.clear → xgrammar_backend.reset → grammar_compiler.clear_cache`。

`finish_reason=length` 的实际生成 tokens 计入 token 吞吐，同时单列截断请求数；它们不是完整合格 SFT 样本。`stop` 也仅表示正常停止，未证明语义正确；正常停止请求速率字段名为 `normal_stop_request_throughput`。当前未测 TTFT，HTTP 总耗时包含排队、grammar、prefill 和 decode。

| 全卡实测结果 | 实际吞吐 | 证据 |
| --- | --- | --- |
| CVM1 8 卡、4×TP2、每副本 c16 池吞吐 | 808.20 输出 tok/s | [64/64 请求、0 API 错误](cvm1_pool_tp2_c16/summary.json) |
| CVM2 8 卡、4×TP2、每副本 c16 池吞吐 | 777.27 输出 tok/s | [64/64 请求、0 API 错误](cvm2_pool_tp2_c16/summary.json) |
| 双 CVM 16 卡联合吞吐 | **1555.08 输出 tok/s**，校时包络 1553.50–1556.67 | [联合结果](joint_16gpu_results.json) |

双机正式测量共同起跑于 2026-10-03 14:52:13 UTC，累计实际输出 244,040 tokens，按校时后的共同窗口约 156.93 秒结算。128 请求均返回、0 API 错误，其中 95 正常停止、33 触及 2048 token 上限。该截断比例是固定测速负载的结果，不能称 128 条合格标注。测试期间两机各自都记录到八张 GPU 同时 100% 利用率；测试后所有八个服务 health=200。测速结束后，服务保持驻留并等待经过质量验收的任务，不会自动启动全量采集或学生训练。

池测量由 [pool_benchmark.py](pool_benchmark.py) 控制：全部副本先通过 health/models/BF16/TP 检查，每副本独立 warmup 4 请求，等待全部 warmup 完成后 flush，再写 `ready.json` 等待 release。两机均 ready 后，操作端在两机原子发布同一未来 epoch；每机结果目录中的 `release.json` 记录实际启动目标时间。每副本测同一批 16 个请求，故每主机 64 个、两机 128 个测量请求，另有 32 个 warmup 请求；这是固定上限的一次诊断，不是持续采集任务。

单机池速率为总实际输出 tokens 除以该机最早客户端请求开始至最晚响应结束的窗口。双机使用 [combine_pools.py](combine_pools.py)，先确认两份 complete 及 workload/harness/TP/concurrency 一致，再合并实际 tokens 和共同时间窗口。`clock_offset_before_pool.json`、`clock_offset_after_pool.json` 中偏移定义为 **CVM2 minus CVM1**，转换 CVM2 时间须减去偏移；上下界进入联合耗时与速率范围。两机 tok/s 不直接相加。两次校时之间时钟偏移保持在记录包络内是该误差范围的假设。

从已有 SSH 配置登录任意一台主机即可查看服务；未配置别名时使用上表的 `ubuntu@IP`，沿用已配置的身份认证。

```bash
ssh cvm-1
```

或：

```bash
ssh cvm-2
```

登录后执行只读检查：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
for port in 31038 31039 31042 31043; do
  curl --fail --silent --show-error --max-time 5 -o /dev/null -w "port ${port}: HTTP %{http_code}\n" "http://127.0.0.1:${port}/health"
done
curl --fail --silent --show-error --max-time 5 http://127.0.0.1:31038/v1/models
```

服务日志、启动和停止回执位于 `/data/caiqingran/sglang_teacher_20261003/runs/<run>/server.log`、`launch.json`、`stop.json`。例如 CVM1 的 GPU0,1 副本：

```bash
task_root=/data/caiqingran/sglang_teacher_20261003
task_run="$task_root/runs/tp2_g01_p31038_v2"
cat "$task_run/launch.json"
tail -n 80 "$task_run/server.log"
```

各机 benchmark 输出在 `task_root/benchmarks/<run>/`：`summary.json`、`run_manifest.json`、`ready.json`、`release.json`；池内每个 `replica_XX/` 包含逐请求的 `*_responses.jsonl` 与客户端 epoch 记录。原始响应和诊断文件按 0600、结果目录按 0700 保存；汇总使用 summary，原始输出不等于训练可用数据。仓库此目录下的同名 summary 是本地归档。

单副本维护按本机 `launch.json` 所有权操作。先结束该副本所属的有限 benchmark，并确认它没有在途请求；下面以 CVM1 GPU0,1/31038 为例，停止脚本会检查 PID 的启动 ticks 和进程组身份，仅对这一 owned 进程组发信号，并写停止回执。

```bash
task_root=/data/caiqingran/sglang_teacher_20261003
task_run="$task_root/runs/tp2_g01_p31038_v2"
python3 "$task_root/stop_replica.py" --run "$task_run"
nvidia-smi -i 0,1 --query-compute-apps=pid,process_name --format=csv
```

两张卡上的该副本进程退出后，使用一个**新的 run 目录**重启。以下命令保持当前 TP2 配置，启动器会检查 GPU 空闲、同任务副本的 GPU 预留及 localhost 端口占用，并拒绝复用已有 `launch.json` 的 run。

```bash
task_root=/data/caiqingran/sglang_teacher_20261003
task_model=/data/caiqingran/models/models/Qwen3.8-27B
task_new_run="$task_root/runs/tp2_g01_p31038_restart_$(date -u +%Y%m%dT%H%M%SZ)"
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 \
python3 "$task_root/launch_replica.py" \
  --python "$task_root/runtime/bin/python" \
  --model "$task_model" --gpus 0,1 --port 31038 \
  --run "$task_new_run" --memory 0.85 --chunk 4096 --concurrency 16 \
  --attention-backend flashinfer
tail -n 80 "$task_new_run/server.log"
curl --fail --silent --show-error --max-time 5 -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:31038/health
```

CVM2 重启时将 `task_model` 改为 `/data/caiqingran/models/Qwen3.8-27B`；其他副本按端口/GPU 表选择对应 run。模型加载与首次编译结束后 `/health` 应返回 200，再核 `/v1/models`。启动回执中的 `health_verified=false` 是启动时状态，不会自动替代后续 health 检查。
