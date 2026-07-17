# vLLM-FL Ascend Custom Ops Tests

本目录包含 `vllm-plugin-FL` 在 Ascend NPU 上的自定义算子连接测试。

## 1. `csrc/ascend/` 下的两种算子接入方式

`csrc/ascend/` 目前承载两类 Ascend 自定义算子，它们的编译、安装和运行加载方式完全不同：

### 1.1 CANN Framework 算子（aclnn 路径）

- **源码位置**：`csrc/ascend/<category>/<op_name>/`
  - 例如 `csrc/ascend/moe/causal_conv1d/`、`csrc/ascend/attention/fused_gdn_gating/`。
- **构建工具链**：CANN `op_host` / `op_kernel` / `aclnn` 工具链。
- **产物**：自解压 `.run` 算子包。
- **安装位置**：默认隔离安装到项目目录 `vllm_fl/_cann_ops_custom/vendors/custom_transformer/`，不污染系统 CANN。
- **运行时加载**：
  - C++ torch extension `vllm_fl._C_ascend` 注册 `torch.ops._C_ascend.*` schema；
  - 算子实现通过 `libopapi.so` + `libcust_opapi.so` 被 CANN 运行时解析；
  - 必须 `source vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash` 设置 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH`。

### 1.2 PTO GDN 预编译算子（Bisheng 路径）

- **源码位置**：`csrc/ascend/pto_chunk_gdn/`。
- **构建工具链**：Bisheng C++ 编译器（`-xcce --cce-aicore-arch=dav-c220`），直接编译 `.cpp` 为 AI Core `.so`。
- **依赖头库**：`csrc/ascend/third_party/pto-isa/`。
- **产物**：多个 `mega_kernel_H*_Hg*_D*_C*.so`。
- **安装位置**：
  - 预编译模式：安装到 Python site-packages 下的 `vllm_fl/ops/pto_chunk_gdn/kernels/compiled_lib/`；
  - JIT 模式：首次调用时由 `vllm_fl/ops/pto_chunk_gdn/compile.py` 自动编译并缓存到同一目录。
- **运行时加载**：Python 代码通过 `ctypes.CDLL` / `torch.ops.load_library` 直接加载 `.so`，不经过 CANN `opp/vendors` 路径。

## 2. 完整编译、安装与接入流程

### 2.1 环境准备

```bash
# 1. 激活 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 确认环境变量
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export SOC_VERSION=ascend910b1   # 根据实际芯片调整
```

### 2.2 编译并安装 torch extension `vllm_fl._C_ascend`

该 extension 把 C++ 算子实现注册到 `torch.ops._C_ascend`，同时包含 `camem_allocator` 等基础设施。

```bash
cd /workspace/vllm-plugin-FL
VLLM_VENDOR=ascend python setup.py build_ext --inplace
```

完成后会在项目根目录生成：

```text
vllm_fl/_C_ascend.cpython-311-aarch64-linux-gnu.so
vllm_fl/libvllm_fl_kernels.so
```

测试脚本中通过 `import vllm_fl._C_ascend` 加载 extension，随后即可调用 `torch.ops._C_ascend.*`。

### 2.3 编译并部署 CANN framework 算子包

```bash
cd /workspace/vllm-plugin-FL/csrc/ascend

# 编译并打包所有 910B 支持的 framework 算子
bash build_aclnn.sh ascend910b
```

`build_aclnn.sh` 内部会：

1. 调用 `bash build.sh --pkg --ops="..." --soc=ascend910b`；
2. 在 `csrc/ascend/build/` 下生成 `cann-ops-transformer-custom_linux-aarch64.run`；
3. 自动执行 `.run --install-path=/workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom`；
4. 最终目录结构：

```text
vllm_fl/_cann_ops_custom/vendors/custom_transformer/
├── bin/set_env.bash          # 设置 ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH
├── op_api/include/aclnnop/   # aclnn 头文件
├── op_api/lib/libcust_opapi.so
├── op_proto/
└── op_impl/
```

每次运行测试前必须先 source 环境脚本：

```bash
source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
```

### 2.4 PTO GDN 算子的两种使用方式

#### 方式 A：预编译（推荐生产环境）

```bash
cd /workspace/vllm-plugin-FL
VLLM_VENDOR=ascend BUILD_PTO_CHUNK_GDN=ON python setup.py build_ext --inplace

# 显式编译 PTO megakernel
cmake --build build/temp.linux-aarch64-cpython-311 \
      --target pto_chunk_gdn_kernels -j$(nproc)
```

产物会安装到：

```text
/usr/local/python3.11.14/lib/python3.11/site-packages/vllm_fl/ops/pto_chunk_gdn/kernels/compiled_lib/
├── mega_kernel_H16_Hg8_D128_C128.so
├── mega_kernel_H16_Hg16_D128_C128.so
└── ...
```

> 在 editable install（`pip install -e .`）下，`_PACKAGE_ROOT` 等于仓库根目录，因此也会写到仓库内的 `vllm_fl/ops/pto_chunk_gdn/kernels/compiled_lib/`。

#### 方式 B：JIT 首次编译（开发调试用）

不预编译，直接运行 `tests/custom_ops_tests/test_pto_chunk_gdn.py`。`vllm_fl/ops/pto_chunk_gdn/compile.py` 会：

1. 自动查找 `csrc/ascend/third_party/pto-isa`；
2. 调用系统 `bisheng` 编译对应配置的 `mega_kernel_*.so`；
3. 缓存到 `vllm_fl/ops/pto_chunk_gdn/kernels/compiled_lib/`；
4. 后续调用直接复用缓存。

### 2.5 目录结构总览

```text
csrc/ascend/
├── CMakeLists.txt              # 构建 _C_ascend + 分发 pto_chunk_gdn
├── torch_binding.cpp           # torch.ops._C_ascend 注册
├── torch_binding_meta.cpp      # meta kernel 注册
├── camem_allocator.cpp         # NPU 显存分配器
├── build.sh                    # CANN framework 算子构建脚本
├── build_aclnn.sh              # 一键打包 + 安装 .run
├── <category>/<op_name>/       # CANN framework 算子源码
│   └── op_host/op_kernel/...
├── pto_chunk_gdn/              # PTO GDN megakernel 源码
│   ├── CMakeLists.txt
│   ├── mega_kernel.cpp
│   └── include/
└── third_party/
    ├── catlass/                # CANN 算子依赖
    └── pto-isa/                # PTO 算子依赖
```

## 3. 如何执行测试

所有测试脚本都需要先 source CANN 自定义算子环境（用于 framework 算子和 torch extension）：

```bash
cd /workspace/vllm-plugin-FL
source vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
```

### 3.1 逐个运行

```bash
# CANN framework 算子
python tests/custom_ops_tests/test_causal_conv1d.py
python tests/custom_ops_tests/test_fused_gdn_gating.py
python tests/custom_ops_tests/test_gemma_rms_norm.py
python tests/custom_ops_tests/test_recurrent_gated_delta_rule.py
python tests/custom_ops_tests/test_chunk_gated_delta_rule_fwd_h.py

# PTO GDN 算子（首次运行会触发 Bisheng JIT 编译）
python tests/custom_ops_tests/test_pto_chunk_gdn.py
```

### 3.2 批量运行

```bash
for f in tests/custom_ops_tests/test_*.py; do
    echo "=== $f ==="
    python "$f" 2>&1 | tail -3
done
```

### 3.3 常见失败原因

| 现象 | 原因 | 解决 |
|---|---|---|
| `AttributeError: '_OpNamespace' '_C_ascend' object has no attribute 'xxx'` | `vllm_fl._C_ascend` 未编译或算子未注册 | 重新执行 `VLLM_VENDOR=ascend python setup.py build_ext --inplace` |
| `aclnnXxx ... not in libopapi.so` | 未 source 自定义算子环境 | `source vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash` |
| `ImportError: dynamic module does not define module export function (PyInit__C_ascend)` | `camem_allocator.cpp` 里的 PyInit 函数名与 extension 名不匹配 | 检查 `csrc/ascend/camem_allocator.cpp` 是否为 `PyInit__C_ascend` |
| PTO 测试提示找不到 `pto-isa` | 子模块未初始化或路径错误 | `git submodule update --init --recursive csrc/ascend/third_party/pto-isa` |

## 4. 测试脚本说明

| 测试脚本 | 对应算子 | 接入方式 |
|---|---|---|
| `test_causal_conv1d.py` | `npu_causal_conv1d_custom` | CANN framework |
| `test_fused_gdn_gating.py` | `npu_fused_gdn_gating` | CANN framework |
| `test_gemma_rms_norm.py` | `npu_gemma_rms_norm` | CANN framework |
| `test_recurrent_gated_delta_rule.py` | `npu_recurrent_gated_delta_rule` | CANN framework |
| `test_chunk_gated_delta_rule_fwd_h.py` | `chunk_gated_delta_rule_fwd_h` | CANN framework |
| `test_pto_chunk_gdn.py` | PTO GDN megakernel | Bisheng PTO |
