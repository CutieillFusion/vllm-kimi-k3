<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## Boot Kimi-K3 on four NVIDIA DGX Sparks

This fork can serve the converted Kimi-K3 1-bit model across four DGX Spark
systems with one GB10 GPU per node. The validated topology is TP=2, PP=2, and
expert parallelism across all four GPUs. All 896 experts remain resident and
the FP8 DS-MLA KV cache supports the full 1,048,576-token context length.

### 1. Prepare every Spark

The four nodes must be able to reach each other over the same private network.
The examples below call them `spark1` through `spark4`, with `spark1` as the
Ray head and OpenAI API server.

Install the same checkout and Python environment at the same path on every
node. Building vLLM on a Spark takes about 25-30 minutes, so keep the checkout
on persistent storage rather than `/tmp`.

```bash
git clone --branch cuda-k3-fp8-ds-mla \
  https://github.com/CutieillFusion/vllm-kimi-k3.git ~/k3vllm
cd ~/k3vllm

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
uv pip install -e . --torch-backend=auto
```

Each node needs roughly 121 GB of unified memory available to vLLM. Stop other
GPU workloads before loading the model.

### 2. Place the converted model on every node

Set `MODEL` to a local path visible at the same location on every Spark. The
model must contain the dense checkpoint and all four converted expert stores:

```text
Kimi-K3-w1/
├── config.json
├── model-*.safetensors
├── tokenizer files
└── k3_w1/
    ├── pp0-tp0/experts.w2
    ├── pp0-tp1/experts.w2
    ├── pp1-tp0/experts.w2
    └── pp1-tp1/experts.w2
```

If starting from an official Kimi-K3 checkpoint, convert it once and then copy
the resulting directory to all four nodes:

```bash
cd ~/k3vllm
.venv/bin/python tools/k3_w1/convert_model.py \
  --model /models/Kimi-K3 \
  --output /models/Kimi-K3-w1
```

The conversion requires substantial temporary disk space and produces about
426 GB of expert stores. See
[`docs/features/quantization/k3_w1.md`](docs/features/quantization/k3_w1.md)
for the format and preparation details.

### 3. Configure networking on all four nodes

Choose the private interface used between the Sparks and substitute the real
head IP below. Run this block in every shell that starts Ray or vLLM. Set
`NODE_IP` to that machine's address.

```bash
cd ~/k3vllm
source .venv/bin/activate

export HEAD_IP=192.168.0.10       # private IP of spark1
export NODE_IP=192.168.0.10       # this node's private IP
export CLUSTER_IFACE=enP7s7       # this node's private-network interface

export VLLM_HOST_IP="$NODE_IP"
export MASTER_ADDR="$HEAD_IP"
export GLOO_SOCKET_IFNAME="$CLUSTER_IFACE"
export NCCL_SOCKET_IFNAME="$CLUSTER_IFACE"
export NCCL_IB_DISABLE=1
export VLLM_DISTRIBUTED_TIMEOUT_SECONDS=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
export VLLM_PP_LAYER_PARTITION=47,46
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_memory_monitor_refresh_ms=0

export K3_BITS=4
export K3_GROUP=64
export K3_MLA_BITS=8
export K3_HEAD_BITS=8
export K3_QUANT_EMBED=1
```

`NCCL_IB_DISABLE=1` uses the private Ethernet network and is the most portable
way to bring up the cluster. RoCE can be enabled later after selecting the
correct local RoCEv2 GID index on each node.

### 4. Start Ray

On `spark1`, start the head first:

```bash
.venv/bin/ray stop --force
.venv/bin/ray start --head \
  --node-ip-address="$NODE_IP" \
  --port=6379 \
  --num-gpus=1 \
  --num-cpus=0 \
  --disable-usage-stats \
  --object-store-memory=200000000 \
  --include-dashboard=false
```

Then run this on `spark2`, `spark3`, and `spark4`, with the correct `NODE_IP`
exported on each machine:

```bash
.venv/bin/ray stop --force
.venv/bin/ray start \
  --address="$HEAD_IP:6379" \
  --node-ip-address="$NODE_IP" \
  --num-gpus=1 \
  --num-cpus=0 \
  --object-store-memory=200000000
```

Confirm that the head sees four live nodes:

```bash
.venv/bin/ray status
```

### 5. Start Kimi-K3 on spark1

The following is the validated full-residency configuration with a 1M-token
FP8 DS-MLA KV cache:

```bash
MODEL=/models/Kimi-K3-w1

.venv/bin/vllm serve "$MODEL" \
  --served-model-name kimi-k3 \
  --quantization k3_w1 \
  --trust-remote-code \
  --reasoning-parser kimi_k3 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --expert-placement-strategy round_robin \
  --distributed-executor-backend ray \
  --max-model-len 1048576 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --gpu-memory-utilization 0.918 \
  --kv-cache-memory-bytes 9395240960 \
  --kv-cache-dtype fp8_ds_mla \
  --no-enable-prefix-caching \
  --enforce-eager \
  --kernel-config '{"enable_flashinfer_autotune":false,"enable_cutedsl_warmup":false,"enable_jit_warmup":false}' \
  --disable-custom-all-reduce \
  --host 0.0.0.0 \
  --port 8000
```

Initial loading takes several minutes. A successful start reports 46/46 MoE
layers filled on every rank and approximately 1,076,763 KV-cache tokens.

Verify the server from another machine:

```bash
curl http://spark1:8000/v1/models

curl http://spark1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kimi-k3",
    "messages": [{"role": "user", "content": "Hello from Kimi-K3"}],
    "max_tokens": 128
  }'
```

To shut down the cluster, stop vLLM on `spark1`, then run
`.venv/bin/ray stop --force` on every node.

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
