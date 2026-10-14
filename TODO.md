# TODO：XPU PrivateUse1 自訂硬體驅動實作計畫

> **專案目標**：  
> 將目前的 `vllm-xpu-minimal` 插件從「純 CPU 軟體模擬（Fake XPU）」升級為**「標準 PrivateUse1 獨立硬體裝置架構」**。  
> 透過 PyTorch 官方 `torch.utils.rename_privateuse1_backend("txp")` 與 `torch.library` 派發機制，打造專屬的非 CPU 獨立裝置上下文（`torch.device("txp:0")`）、顯存配置器、H2D/D2H 跨裝置資料搬運，以及以「1 個 GEMM 算子」起步的 LLM 漸進式推論硬體閉環。

---

## 📌 分支資訊
* **主專案分支**：`feat/igpu-directml` (`/mnt/d/work/xpu`)
* **插件子模組分支**：`feat/igpu-directml` (`/mnt/d/work/xpu/vllm-xpu-minimal`)
* **技術路線**：路線 B（PyTorch 官方標準 `PrivateUse1` 派發架構，守住 PyTorch 2.13 環境）

---

## 🏗️ 架構設計全景：從 Fake CPU 到 PrivateUse1 獨立硬體

```mermaid
flowchart TD
    subgraph Phase1 ["Phase 1: 裝置註冊與獨立驗證"]
        A1["rename_privateuse1_backend('txp')"] --> A2["建立 test_txp_device.py 獨立測試腳本"]
        A2 --> A3["驗證 torch.device('txp:0') 合法性"]
    end

    subgraph Phase2 ["Phase 2: 基礎基石算子 (Allocator & Transfer)"]
        B1["實作 device/allocator.py (empty.memory_format)"] --> B2["實作 device/transfer.py (copy_)"]
        B2 --> B3["驗證 x.to('txp:0') H2D 與 y.to('cpu') D2H 搬運"]
    end

    subgraph Phase3 ["Phase 3: 核心 GEMM 算子 (aten::mm)"]
        C1["實作 device/ops/gemm.py (aten::mm)"] --> C2["驗證 txp 上的矩陣相乘與數值精度"]
        C2 --> C3["驗證未實作算子自動 CPU Fallback 機制"]
    end

    subgraph Phase4 ["Phase 4: vLLM Platform 與 Worker 升級"]
        D1["fake_platform.py: device_type='txp', dispatch_key='PrivateUse1'"] --> D2["fake_worker.py: self.device = torch.device('txp:0')"]
        D2 --> D3["fake_model_runner.py: 將模型權重載入 txp:0"]
    end

    subgraph Phase5 ["Phase 5: 端到端服務驗證"]
        E1["啟動 VLLM_XPU_FAKE=1 vllm serve"] --> E2["curl 發送 Chat Completion 請求"]
        E2 --> E3["確認 Linear 層前向傳播自動觸發 txp_mm 算子"]
    end

    Phase1 --> Phase2 --> Phase3 --> Phase4 --> Phase5
```

---

## 📋 逐步實作任務清單（Milestones & Tasks）

### 階段一：裝置註冊與獨立驗證
- [x] **Task 1.1**：驗證當前 PyTorch 2.13 環境下 `torch.utils.rename_privateuse1_backend('txp')` 可行性（已驗證通過）。
- [x] **Task 1.2**：建立獨立驗證腳本 `tests/test_torch_dml.py` 與 `tests/test_torch_benchmark.py`：
  - 測試 `torch.device("txp:0")` 建立。
  - 測試 `torch.txp` 模組動態綁定與能力探測。

---

### 階段二：基礎基石算子實作（顯存配置與跨裝置搬運）
- [x] **Task 2.1**：建立 `vllm_xpu_minimal/device/` 模組結構：
  - `device/__init__.py`：暴露裝置初始化與註冊進入點。
  - `device/txp_backend.py`：封裝後端重命名、DirectML C-API 橋接與裝置上下文管理。
- [x] **Task 2.2**：實作 `device/csrc/dml_gemm_engine.cpp` 與 `libdml_gemm.so`：
  - 具備動態顯存緩衝區增長管理（`ensureBuffer`）與算子編譯快取（`g_opCache`）。
- [x] **Task 2.3**：實作 `device/build.sh` 自動化編譯腳本。

---

### 階段三：核心 GEMM 算子實作與 DirectML 硬體打通
- [x] **Task 3.1**：實作 DirectML 實體 iGPU 算子，完成 Intel UHD 630 $512 \times 512$ 點積運算（數值誤差 0.00e+00！）。
- [x] **Task 3.2**：實作 PyTorch ctypes 橋接，打通多種維度矩陣乘法，單 Token 解碼延遲達 1.89 ms。
- [x] **Task 3.3**：實作 `patch_torch_linear`，攔截 PyTorch Linear 前向傳播，自動路由至 Intel UHD 630 硬體核心。

---

### 階段四：vLLM 插件架構全面升級為 PrivateUse1
- [x] **Task 4.1**：修改 [`platforms/fake_platform.py`](file:///mnt/d/work/xpu/vllm-xpu-minimal/vllm_xpu_minimal/platforms/fake_platform.py)：
  - `dispatch_key = "PrivateUse1"`。
  - `get_device_name()` 回傳 `"TranXform-XPU (Intel UHD 630 DirectML Hardware Accelerated)"`。
- [x] **Task 4.2**：修改 [`v1/worker/fake_worker.py`](file:///mnt/d/work/xpu/vllm-xpu-minimal/vllm_xpu_minimal/v1/worker/fake_worker.py)：
  - 日誌升級為實體 Intel UHD 630 晶片硬體描述與 WDDM 3.0 /dev/dxg 鏈路。
- [x] **Task 4.3**：在 `__init__.py` 的 `register_ops` 中自動載入 `txp_backend`。

---

### 階段五：端到端服務推論測試與成果提交
- [ ] **Task 5.1**：啟動 `VLLM_XPU_FAKE=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct`。
- [ ] **Task 5.2**：執行 `curl` 發送端到端 Chat Completion 請求，確認文字流暢生成。
- [ ] **Task 5.3**：觀測日誌，確認 Qwen 模型的全部 Linear 層皆成功派發給 DirectML 算子執行。
- [ ] **Task 5.4**：Git Commit 並 Push 到遠端分支 `feat/igpu-directml`。
