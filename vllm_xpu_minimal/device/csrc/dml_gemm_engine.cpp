#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <thread>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <dlfcn.h>
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include <directx/dxcore.h>
#include <directx/DirectML.h>

// 官方精準 GUID 定義
static const GUID IID_IDXCoreAdapterFactory_val  = { 0x78ee5945, 0xc36e, 0x4b13, { 0xa6, 0x69, 0x00, 0x5d, 0xd1, 0x1c, 0x0f, 0x06 } };
static const GUID IID_IDXCoreAdapterList_val     = { 0x526c7776, 0x40e9, 0x459b, { 0xb7, 0x11, 0xf3, 0x2a, 0xd7, 0x6d, 0xfc, 0x28 } };
static const GUID IID_IDXCoreAdapter_val         = { 0xf0db4c7f, 0xfe5a, 0x42a2, { 0xbd, 0x62, 0xf2, 0xa6, 0xcf, 0x6f, 0xc8, 0x3e } };
static const GUID IID_ID3D12Device_val           = { 0x189819f1, 0x1db6, 0x4b57, { 0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7 } };
static const GUID IID_ID3D12CommandQueue_val     = { 0x0ec870a6, 0x5d7e, 0x4c22, { 0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed } };
static const GUID IID_ID3D12CommandAllocator_val = { 0x6102dee4, 0xaf59, 0x4b09, { 0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24 } };
static const GUID IID_ID3D12GraphicsCommandList_val = { 0x5b160d0f, 0xac1b, 0x4185, { 0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55 } };
static const GUID IID_ID3D12Fence_val            = { 0x0a753dcf, 0xc4d8, 0x4b91, { 0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76 } };
static const GUID IID_ID3D12DescriptorHeap_val   = { 0x8efb471d, 0x616c, 0x4f49, { 0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51 } };
static const GUID IID_ID3D12Resource_val         = { 0x696442be, 0xa72e, 0x4059, { 0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad } };
static const GUID IID_IDMLDevice_val             = { 0x6dbd6437, 0x96fd, 0x423f, { 0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f } };
static const GUID IID_IDMLOperator_val           = { 0x26caae7a, 0x3081, 0x4633, { 0x95, 0x81, 0x22, 0x6f, 0xbe, 0x57, 0x69, 0x5d } };
static const GUID IID_IDMLCompiledOperator_val   = { 0x6b15e56a, 0xbf5c, 0x4902, { 0x92, 0xd8, 0xda, 0x3a, 0x65, 0x0a, 0xfe, 0xa4 } };
static const GUID IID_IDMLOperatorInitializer_val= { 0x427c1113, 0x435c, 0x469c, { 0x86, 0x76, 0x4d, 0x5d, 0xd0, 0x72, 0xf8, 0x13 } };
static const GUID IID_IDMLCommandRecorder_val    = { 0xe6857a76, 0x2e3e, 0x4fdd, { 0xbf, 0xf4, 0x5d, 0x2b, 0xa1, 0x0f, 0xb4, 0x53 } };
static const GUID IID_IDMLBindingTable_val       = { 0x29c687dc, 0xde74, 0x4e3b, { 0xab, 0x00, 0x11, 0x68, 0xf2, 0xfc, 0x3c, 0xfc } };

typedef HRESULT (*DXCoreCreateAdapterFactoryFn)(REFIID riid, void** ppv);
typedef HRESULT (*D3D12CreateDeviceFn)(IUnknown* pAdapter, D3D_FEATURE_LEVEL MinimumFeatureLevel, REFIID riid, void** ppv);
typedef HRESULT (*DMLCreateDeviceFn)(ID3D12Device* d3d12Device, DML_CREATE_DEVICE_FLAGS flags, REFIID riid, void** ppv);

// 全域持久硬體狀態
static bool g_initialized = false;
static IDXCoreAdapterFactory* g_factory = nullptr;
static IDXCoreAdapter* g_adapter = nullptr;
static ID3D12Device* g_d3d_device = nullptr;
static ID3D12CommandQueue* g_commandQueue = nullptr;
static ID3D12CommandAllocator* g_commandAllocator = nullptr;
static ID3D12GraphicsCommandList* g_commandList = nullptr;
static ID3D12Fence* g_fence = nullptr;
static uint64_t g_fenceValue = 0;
static IDMLDevice* g_dml_device = nullptr;
static IDMLCommandRecorder* g_commandRecorder = nullptr;

struct CachedOp {
    IDMLCompiledOperator* compiledOp = nullptr;
    ID3D12DescriptorHeap* descriptorHeap = nullptr;
    IDMLBindingTable* bindingTable = nullptr;
    uint32_t descriptorCount = 0;
};

static std::map<std::tuple<uint32_t, uint32_t, uint32_t>, CachedOp> g_opCache;

// ── Stage 0: 效能剖析計數器 ──
static uint64_t g_gemmCount = 0;
static double g_gemmH2DMs = 0.0;
static double g_gemmDispatchMs = 0.0;
static double g_gemmD2HMs = 0.0;
static double g_gemmFenceMs = 0.0;

static double nowMs() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ── Stage 1: 裝置常駐張量池（default heap）──
// dtype: 0 = FLOAT32, 1 = FLOAT16
struct TensorEntry {
    ID3D12Resource* resource = nullptr;
    uint32_t bytes = 0;
    uint32_t dtype = 0;
    bool in_use = false;
};

static std::vector<TensorEntry> g_tensors;
static std::vector<uint64_t> g_freeTensorIds;

static size_t dtypeSize(uint32_t dtype) {
    return dtype == 1 ? 2 : 4;
}

static DML_TENSOR_DATA_TYPE dmlDtype(uint32_t dtype) {
    return dtype == 1 ? DML_TENSOR_DATA_TYPE_FLOAT16 : DML_TENSOR_DATA_TYPE_FLOAT32;
}

// Stage 1: handle-based op cache（M, K, N, dtype, fused_activation）
static std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, int>, CachedOp> g_handleOpCache;
// Stage 1: elementwise mul op cache（count, dtype）
static std::map<std::tuple<uint32_t, uint32_t>, CachedOp> g_mulCache;
// Stage 1: sigmoid op cache（count, dtype）
static std::map<std::tuple<uint32_t, uint32_t>, CachedOp> g_sigmoidCache;

// 緩衝區緩存 (動態增長，避免反覆配置顯存)
static ID3D12Resource* g_uploadA = nullptr;
static size_t g_uploadASize = 0;
static ID3D12Resource* g_inputA = nullptr;
static size_t g_inputASize = 0;

static ID3D12Resource* g_uploadB = nullptr;
static size_t g_uploadBSize = 0;
static ID3D12Resource* g_inputB = nullptr;
static size_t g_inputBSize = 0;

static ID3D12Resource* g_outputC = nullptr;
static size_t g_outputCSize = 0;
static ID3D12Resource* g_readbackC = nullptr;
static size_t g_readbackCSize = 0;

static void CloseExecuteResetWait() {
    double t0 = nowMs();
    g_commandList->Close();
    ID3D12CommandList* lists[] = { g_commandList };
    g_commandQueue->ExecuteCommandLists(1, lists);

    g_fenceValue++;
    g_commandQueue->Signal(g_fence, g_fenceValue);
    while (g_fence->GetCompletedValue() < g_fenceValue) {
        std::this_thread::yield();
    }
    g_commandAllocator->Reset();
    g_commandList->Reset(g_commandAllocator, nullptr);
    g_gemmFenceMs += nowMs() - t0;
}

static D3D12_RESOURCE_DESC makeBufferDesc(size_t size, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

static void ensureBuffer(ID3D12Resource*& buf, size_t& currentSize, size_t neededSize, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState, D3D12_RESOURCE_FLAGS flags) {
    if (buf && currentSize >= neededSize) return;

    if (buf) {
        buf->Release();
        buf = nullptr;
    }

    currentSize = neededSize;
    D3D12_HEAP_PROPERTIES heapProps = { heapType, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    auto desc = makeBufferDesc(neededSize, flags);
    g_d3d_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc, initialState, nullptr, IID_ID3D12Resource_val, (void**)&buf);
}

// ── Stage 1: 通用算子編譯與派發 helper ──
static CachedOp compileOperator(DML_OPERATOR_DESC* opDesc) {
    CachedOp result = { nullptr, nullptr, nullptr, 0 };

    IDMLOperator* dmlOperator = nullptr;
    HRESULT hr = g_dml_device->CreateOperator(opDesc, IID_IDMLOperator_val, (void**)&dmlOperator);
    if (FAILED(hr)) {
        fprintf(stderr, "[dml_gemm] CreateOperator failed: 0x%08x\n", (unsigned)hr);
        return result;
    }

    IDMLCompiledOperator* compiledOp = nullptr;
    hr = g_dml_device->CompileOperator(dmlOperator, DML_EXECUTION_FLAG_NONE, IID_IDMLCompiledOperator_val, (void**)&compiledOp);
    if (FAILED(hr)) {
        fprintf(stderr, "[dml_gemm] CompileOperator failed: 0x%08x\n", (unsigned)hr);
        dmlOperator->Release();
        return result;
    }
    dmlOperator->Release();

    IDMLOperatorInitializer* initializer = nullptr;
    IDMLCompiledOperator* ops[] = { compiledOp };
    hr = g_dml_device->CreateOperatorInitializer(1, ops, IID_IDMLOperatorInitializer_val, (void**)&initializer);
    if (FAILED(hr)) {
        fprintf(stderr, "[dml_gemm] CreateOperatorInitializer failed: 0x%08x\n", (unsigned)hr);
        compiledOp->Release();
        return result;
    }

    DML_BINDING_PROPERTIES initProps = initializer->GetBindingProperties();
    DML_BINDING_PROPERTIES execProps = compiledOp->GetBindingProperties();
    uint32_t descriptorCount = std::max(initProps.RequiredDescriptorCount, execProps.RequiredDescriptorCount);

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = descriptorCount;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    ID3D12DescriptorHeap* descriptorHeap = nullptr;
    hr = g_d3d_device->CreateDescriptorHeap(&heapDesc, IID_ID3D12DescriptorHeap_val, (void**)&descriptorHeap);
    if (FAILED(hr)) {
        fprintf(stderr, "[dml_gemm] CreateDescriptorHeap failed: 0x%08x\n", (unsigned)hr);
        initializer->Release();
        compiledOp->Release();
        return result;
    }

    DML_BINDING_TABLE_DESC tableDesc = {};
    tableDesc.Dispatchable = initializer;
    tableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    tableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    tableDesc.SizeInDescriptors = descriptorCount;

    IDMLBindingTable* bindingTable = nullptr;
    hr = g_dml_device->CreateBindingTable(&tableDesc, IID_IDMLBindingTable_val, (void**)&bindingTable);
    if (FAILED(hr)) {
        fprintf(stderr, "[dml_gemm] CreateBindingTable failed: 0x%08x\n", (unsigned)hr);
        descriptorHeap->Release();
        initializer->Release();
        compiledOp->Release();
        return result;
    }

    if (initProps.RequiredDescriptorCount > 0 || initProps.PersistentResourceSize > 0) {
        ID3D12DescriptorHeap* heaps[] = { descriptorHeap };
        g_commandList->SetDescriptorHeaps(1, heaps);
        g_commandRecorder->RecordDispatch(g_commandList, initializer, bindingTable);
        CloseExecuteResetWait();
    }
    initializer->Release();

    result.compiledOp = compiledOp;
    result.descriptorHeap = descriptorHeap;
    result.bindingTable = bindingTable;
    result.descriptorCount = descriptorCount;
    return result;
}

static int dispatchOperator(CachedOp& op, DML_BINDING_DESC* inputDescs, uint32_t numInputs, DML_BINDING_DESC* outputDesc) {
    ID3D12DescriptorHeap* heaps[] = { op.descriptorHeap };
    g_commandList->SetDescriptorHeaps(1, heaps);

    DML_BINDING_TABLE_DESC tableDesc = {};
    tableDesc.Dispatchable = op.compiledOp;
    tableDesc.CPUDescriptorHandle = op.descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    tableDesc.GPUDescriptorHandle = op.descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    tableDesc.SizeInDescriptors = op.descriptorCount;
    op.bindingTable->Reset(&tableDesc);

    op.bindingTable->BindInputs(numInputs, inputDescs);

    op.bindingTable->BindOutputs(1, outputDesc);

    g_commandRecorder->RecordDispatch(g_commandList, op.compiledOp, op.bindingTable);
    CloseExecuteResetWait();
    return 0;
}

extern "C" {

int dml_gemm_init() {
    if (g_initialized) return 0;

    void* dxcore_lib = dlopen("/usr/lib/wsl/lib/libdxcore.so", RTLD_NOW);
    void* d3d12_lib  = dlopen("/usr/lib/wsl/lib/libd3d12.so", RTLD_NOW);
    void* dml_lib    = dlopen("/mnt/d/work/xpu/vllm-xpu-minimal/vllm_xpu_minimal/device/lib/libdirectml.so", RTLD_NOW);

    if (!dxcore_lib || !d3d12_lib || !dml_lib) return -1;

    auto DXCoreCreateAdapterFactory_ptr = (DXCoreCreateAdapterFactoryFn)dlsym(dxcore_lib, "DXCoreCreateAdapterFactory");
    auto D3D12CreateDevice_ptr = (D3D12CreateDeviceFn)dlsym(d3d12_lib, "D3D12CreateDevice");
    auto DMLCreateDevice_ptr = (DMLCreateDeviceFn)dlsym(dml_lib, "DMLCreateDevice");

    DXCoreCreateAdapterFactory_ptr(IID_IDXCoreAdapterFactory_val, (void**)&g_factory);

    static const GUID DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE_val = { 0x248e2800, 0xa793, 0x4724, { 0xab, 0xaa, 0x23, 0xa6, 0xde, 0x1b, 0xe0, 0x90 } };
    IDXCoreAdapterList* adapterList = nullptr;
    const GUID filterGuids[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE_val };
    g_factory->CreateAdapterList(1, filterGuids, IID_IDXCoreAdapterList_val, (void**)&adapterList);
    adapterList->GetAdapter(0, IID_IDXCoreAdapter_val, (void**)&g_adapter);
    adapterList->Release();

    D3D12CreateDevice_ptr(g_adapter, D3D_FEATURE_LEVEL_11_0, IID_ID3D12Device_val, (void**)&g_d3d_device);

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    g_d3d_device->CreateCommandQueue(&queueDesc, IID_ID3D12CommandQueue_val, (void**)&g_commandQueue);
    g_d3d_device->CreateCommandAllocator(queueDesc.Type, IID_ID3D12CommandAllocator_val, (void**)&g_commandAllocator);
    g_d3d_device->CreateCommandList(0, queueDesc.Type, g_commandAllocator, nullptr, IID_ID3D12GraphicsCommandList_val, (void**)&g_commandList);

    g_d3d_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence_val, (void**)&g_fence);
    g_fenceValue = 0;

    DMLCreateDevice_ptr(g_d3d_device, DML_CREATE_DEVICE_FLAG_NONE, IID_IDMLDevice_val, (void**)&g_dml_device);
    g_dml_device->CreateCommandRecorder(IID_IDMLCommandRecorder_val, (void**)&g_commandRecorder);

    g_initialized = true;
    return 0;
}

int dml_gemm_forward(const float* a, const float* b, float* c, uint32_t M, uint32_t K, uint32_t N) {
    if (!g_initialized) {
        if (dml_gemm_init() != 0) return -1;
    }

    double t0 = nowMs();

    size_t aBufferSize = (size_t)M * K * sizeof(float);
    size_t bBufferSize = (size_t)K * N * sizeof(float);
    size_t cBufferSize = (size_t)M * N * sizeof(float);

    // 1. 查找或編譯算子快取
    auto shapeKey = std::make_tuple(M, K, N);
    auto it = g_opCache.find(shapeKey);
    if (it == g_opCache.end()) {
        uint32_t aSizes[] = { 1, 1, M, K };
        uint32_t bSizes[] = { 1, 1, K, N };
        uint32_t cSizes[] = { 1, 1, M, N };

        DML_BUFFER_TENSOR_DESC aTensorBufferDesc = {};
        aTensorBufferDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
        aTensorBufferDesc.Flags = DML_TENSOR_FLAG_NONE;
        aTensorBufferDesc.DimensionCount = 4;
        aTensorBufferDesc.Sizes = aSizes;
        aTensorBufferDesc.TotalTensorSizeInBytes = aBufferSize;
        DML_TENSOR_DESC aTensorDesc = { DML_TENSOR_TYPE_BUFFER, &aTensorBufferDesc };

        DML_BUFFER_TENSOR_DESC bTensorBufferDesc = {};
        bTensorBufferDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
        bTensorBufferDesc.Flags = DML_TENSOR_FLAG_NONE;
        bTensorBufferDesc.DimensionCount = 4;
        bTensorBufferDesc.Sizes = bSizes;
        bTensorBufferDesc.TotalTensorSizeInBytes = bBufferSize;
        DML_TENSOR_DESC bTensorDesc = { DML_TENSOR_TYPE_BUFFER, &bTensorBufferDesc };

        DML_BUFFER_TENSOR_DESC cTensorBufferDesc = {};
        cTensorBufferDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
        cTensorBufferDesc.Flags = DML_TENSOR_FLAG_NONE;
        cTensorBufferDesc.DimensionCount = 4;
        cTensorBufferDesc.Sizes = cSizes;
        cTensorBufferDesc.TotalTensorSizeInBytes = cBufferSize;
        DML_TENSOR_DESC cTensorDesc = { DML_TENSOR_TYPE_BUFFER, &cTensorBufferDesc };

        DML_GEMM_OPERATOR_DESC gemmDesc = {};
        gemmDesc.ATensor = &aTensorDesc;
        gemmDesc.BTensor = &bTensorDesc;
        gemmDesc.CTensor = nullptr;
        gemmDesc.OutputTensor = &cTensorDesc;
        gemmDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
        gemmDesc.TransB = DML_MATRIX_TRANSFORM_NONE;
        gemmDesc.Alpha = 1.0f;
        gemmDesc.Beta = 0.0f;
        gemmDesc.FusedActivation = nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_GEMM, &gemmDesc };
        IDMLOperator* dmlOperator = nullptr;
        g_dml_device->CreateOperator(&opDesc, IID_IDMLOperator_val, (void**)&dmlOperator);

        IDMLCompiledOperator* compiledOp = nullptr;
        g_dml_device->CompileOperator(dmlOperator, DML_EXECUTION_FLAG_NONE, IID_IDMLCompiledOperator_val, (void**)&compiledOp);
        dmlOperator->Release();

        IDMLOperatorInitializer* initializer = nullptr;
        IDMLCompiledOperator* ops[] = { compiledOp };
        g_dml_device->CreateOperatorInitializer(1, ops, IID_IDMLOperatorInitializer_val, (void**)&initializer);

        DML_BINDING_PROPERTIES initProps = initializer->GetBindingProperties();
        DML_BINDING_PROPERTIES execProps = compiledOp->GetBindingProperties();
        uint32_t descriptorCount = std::max(initProps.RequiredDescriptorCount, execProps.RequiredDescriptorCount);

        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = descriptorCount;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        ID3D12DescriptorHeap* descriptorHeap = nullptr;
        g_d3d_device->CreateDescriptorHeap(&heapDesc, IID_ID3D12DescriptorHeap_val, (void**)&descriptorHeap);

        DML_BINDING_TABLE_DESC tableDesc = {};
        tableDesc.Dispatchable = initializer;
        tableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
        tableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
        tableDesc.SizeInDescriptors = descriptorCount;

        IDMLBindingTable* bindingTable = nullptr;
        g_dml_device->CreateBindingTable(&tableDesc, IID_IDMLBindingTable_val, (void**)&bindingTable);

        if (initProps.RequiredDescriptorCount > 0 || initProps.PersistentResourceSize > 0) {
            ID3D12DescriptorHeap* heaps[] = { descriptorHeap };
            g_commandList->SetDescriptorHeaps(1, heaps);
            g_commandRecorder->RecordDispatch(g_commandList, initializer, bindingTable);
            CloseExecuteResetWait();
        }
        initializer->Release();

        CachedOp cached = { compiledOp, descriptorHeap, bindingTable, descriptorCount };
        g_opCache[shapeKey] = cached;
        it = g_opCache.find(shapeKey);
    }

    CachedOp& op = it->second;

    // 2. 確保顯存與上傳緩衝區大小充足
    ensureBuffer(g_uploadA, g_uploadASize, aBufferSize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    ensureBuffer(g_inputA, g_inputASize, aBufferSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    ensureBuffer(g_uploadB, g_uploadBSize, bBufferSize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    ensureBuffer(g_inputB, g_inputBSize, bBufferSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    ensureBuffer(g_outputC, g_outputCSize, cBufferSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ensureBuffer(g_readbackC, g_readbackCSize, cBufferSize, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);

    // 3. 寫入輸入數據 (H2D)
    float* aMapped = nullptr;
    g_uploadA->Map(0, nullptr, (void**)&aMapped);
    memcpy(aMapped, a, aBufferSize);
    g_uploadA->Unmap(0, nullptr);

    float* bMapped = nullptr;
    g_uploadB->Map(0, nullptr, (void**)&bMapped);
    memcpy(bMapped, b, bBufferSize);
    g_uploadB->Unmap(0, nullptr);

    g_commandList->CopyBufferRegion(g_inputA, 0, g_uploadA, 0, aBufferSize);
    g_commandList->CopyBufferRegion(g_inputB, 0, g_uploadB, 0, bBufferSize);

    D3D12_RESOURCE_BARRIER uploadBarriers[2] = {};
    uploadBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    uploadBarriers[0].Transition.pResource = g_inputA;
    uploadBarriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    uploadBarriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    uploadBarriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    uploadBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    uploadBarriers[1].Transition.pResource = g_inputB;
    uploadBarriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    uploadBarriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
uploadBarriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_commandList->ResourceBarrier(2, uploadBarriers);

double t1 = nowMs();
    CloseExecuteResetWait();
    double t2 = nowMs();

    // 4. 重置 Binding Table 並綁定運算
    ID3D12DescriptorHeap* heaps[] = { op.descriptorHeap };
    g_commandList->SetDescriptorHeaps(1, heaps);

    DML_BINDING_TABLE_DESC tableDesc = {};
    tableDesc.Dispatchable = op.compiledOp;
    tableDesc.CPUDescriptorHandle = op.descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    tableDesc.GPUDescriptorHandle = op.descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    tableDesc.SizeInDescriptors = op.descriptorCount;
    op.bindingTable->Reset(&tableDesc);

    DML_BUFFER_BINDING aBinding = { g_inputA, 0, aBufferSize };
    DML_BUFFER_BINDING bBinding = { g_inputB, 0, bBufferSize };
    DML_BUFFER_BINDING cBinding = { g_outputC, 0, cBufferSize };

    DML_BINDING_DESC inputBindings[] = {
        { DML_BINDING_TYPE_BUFFER, &aBinding },
        { DML_BINDING_TYPE_BUFFER, &bBinding },
        { DML_BINDING_TYPE_NONE, nullptr }
    };
    op.bindingTable->BindInputs(3, inputBindings);

    DML_BINDING_DESC outputBinding = { DML_BINDING_TYPE_BUFFER, &cBinding };
    op.bindingTable->BindOutputs(1, &outputBinding);

    // 5. 派發執行
    g_commandRecorder->RecordDispatch(g_commandList, op.compiledOp, op.bindingTable);
    double t3 = nowMs();
    CloseExecuteResetWait();
    double t4 = nowMs();

    // 6. 讀回 (D2H)
    D3D12_RESOURCE_BARRIER readbackBarrier = {};
    readbackBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    readbackBarrier.Transition.pResource = g_outputC;
    readbackBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    readbackBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    readbackBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_commandList->ResourceBarrier(1, &readbackBarrier);

    g_commandList->CopyResource(g_readbackC, g_outputC);

    readbackBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    readbackBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    g_commandList->ResourceBarrier(1, &readbackBarrier);

    double t5 = nowMs();
    CloseExecuteResetWait();
    double t6 = nowMs();

    D3D12_RANGE readRange = { 0, cBufferSize };
    float* cMapped = nullptr;
    g_readbackC->Map(0, &readRange, (void**)&cMapped);
    memcpy(c, cMapped, cBufferSize);
    g_readbackC->Unmap(0, nullptr);

    double t7 = nowMs();

    // ── Stage 0: 累計剖析統計（fence 由 CloseExecuteResetWait 統一累計）──
    g_gemmCount++;
    g_gemmH2DMs += t1 - t0;
    g_gemmDispatchMs += t3 - t2;
    g_gemmD2HMs += (t5 - t4) + (t7 - t6);

    return 0;
}

// ── Stage 0: 剖析統計查詢 ──
void dml_gemm_get_stats(uint64_t* count, double* h2dMs, double* dispatchMs, double* d2hMs, double* fenceMs) {
    if (count) *count = g_gemmCount;
    if (h2dMs) *h2dMs = g_gemmH2DMs;
    if (dispatchMs) *dispatchMs = g_gemmDispatchMs;
    if (d2hMs) *d2hMs = g_gemmD2HMs;
    if (fenceMs) *fenceMs = g_gemmFenceMs;
}

// ── Stage 1: 裝置常駐張量池 API ──
int dml_tensor_alloc(uint32_t bytes, uint32_t dtype, uint64_t* out_handle) {
    if (!g_initialized) {
        if (dml_gemm_init() != 0) return -1;
    }
    if (!out_handle) return -2;

    // 先找 free list 中可重用的（同 dtype 且容量足夠）
    for (size_t i = 0; i < g_freeTensorIds.size(); i++) {
        uint64_t id = g_freeTensorIds[i];
        TensorEntry& e = g_tensors[id];
        if (e.bytes >= bytes && e.dtype == dtype) {
            e.in_use = true;
            g_freeTensorIds.erase(g_freeTensorIds.begin() + i);
            *out_handle = id;
            return 0;
        }
    }

    TensorEntry e;
    e.bytes = bytes;
    e.dtype = dtype;
    e.in_use = true;

    D3D12_HEAP_PROPERTIES heapProps = { D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    auto desc = makeBufferDesc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (FAILED(g_d3d_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_ID3D12Resource_val, (void**)&e.resource))) {
        return -3;
    }

    g_tensors.push_back(e);
    *out_handle = g_tensors.size() - 1;
    return 0;
}

int dml_tensor_free(uint64_t handle) {
    if (handle >= g_tensors.size()) return -3;
    TensorEntry& t = g_tensors[handle];
    if (!t.in_use) return 0;
    t.in_use = false;
    g_freeTensorIds.push_back(handle);
    return 0;
}

int dml_tensor_upload(uint64_t handle, const void* data, uint32_t bytes) {
    if (!g_initialized) {
        if (dml_gemm_init() != 0) return -1;
    }
    if (handle >= g_tensors.size()) return -3;
    TensorEntry& t = g_tensors[handle];
    if (!t.in_use) return -3;
    if (t.bytes < bytes) return -4;

    double t0 = nowMs();

    ensureBuffer(g_uploadA, g_uploadASize, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);

    void* mapped = nullptr;
    g_uploadA->Map(0, nullptr, &mapped);
    memcpy(mapped, data, bytes);
    g_uploadA->Unmap(0, nullptr);

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = t.resource;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_commandList->ResourceBarrier(1, &barrier);

    g_commandList->CopyBufferRegion(t.resource, 0, g_uploadA, 0, bytes);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    g_commandList->ResourceBarrier(1, &barrier);

    CloseExecuteResetWait();
    g_gemmH2DMs += nowMs() - t0;
    return 0;
}

int dml_tensor_readback(uint64_t handle, void* data, uint32_t bytes) {
    if (!g_initialized) {
        if (dml_gemm_init() != 0) return -1;
    }
    if (handle >= g_tensors.size()) return -3;
    TensorEntry& t = g_tensors[handle];
    if (!t.in_use) return -3;
    if (t.bytes < bytes) return -4;

    double t0 = nowMs();

    ensureBuffer(g_readbackC, g_readbackCSize, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = t.resource;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_commandList->ResourceBarrier(1, &barrier);

    g_commandList->CopyResource(g_readbackC, t.resource);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    g_commandList->ResourceBarrier(1, &barrier);

    CloseExecuteResetWait();

    D3D12_RANGE range = { 0, bytes };
    void* mapped = nullptr;
    g_readbackC->Map(0, &range, &mapped);
    memcpy(data, mapped, bytes);
    g_readbackC->Unmap(0, nullptr);
    g_gemmD2HMs += nowMs() - t0;
    return 0;
}

// ── Stage 1: handle-based GEMM（裝置常駐，支援 dtype）──
// 註：目前 DirectML 版本無 DML_OPERATOR_ACTIVATION_SILU，fused_activation 僅接受 0；
//     silu 由 Python 端以 sigmoid + elementwise_mul 分解（全裝置端，零 CPU 往返）。
int dml_gemm(uint64_t a, uint64_t b, uint64_t c, uint32_t M, uint32_t K, uint32_t N,
             uint32_t dtype, int fused_activation) {
    if (!g_initialized) {
        if (dml_gemm_init() != 0) return -1;
    }
    if (fused_activation != 0) return -6;
    if (a >= g_tensors.size() || b >= g_tensors.size() || c >= g_tensors.size()) return -3;
    TensorEntry& ta = g_tensors[a];
    TensorEntry& tb = g_tensors[b];
    TensorEntry& tc = g_tensors[c];
    if (!ta.in_use || !tb.in_use || !tc.in_use) return -3;

    size_t elemSize = dtypeSize(dtype);
    size_t aBytes = (size_t)M * K * elemSize;
    size_t bBytes = (size_t)K * N * elemSize;
    size_t cBytes = (size_t)M * N * elemSize;
    if (ta.bytes < aBytes || tb.bytes < bBytes || tc.bytes < cBytes) return -4;

    double t0 = nowMs();

    auto shapeKey = std::make_tuple(M, K, N, dtype, fused_activation);
    auto it = g_handleOpCache.find(shapeKey);
    if (it == g_handleOpCache.end()) {
        uint32_t aSizes[] = { 1, 1, M, K };
        uint32_t bSizes[] = { 1, 1, K, N };
        uint32_t cSizes[] = { 1, 1, M, N };

        DML_BUFFER_TENSOR_DESC aTensorBufferDesc = {};
        aTensorBufferDesc.DataType = dmlDtype(dtype);
        aTensorBufferDesc.Flags = DML_TENSOR_FLAG_NONE;
        aTensorBufferDesc.DimensionCount = 4;
        aTensorBufferDesc.Sizes = aSizes;
        aTensorBufferDesc.TotalTensorSizeInBytes = aBytes;
        DML_TENSOR_DESC aTensorDesc = { DML_TENSOR_TYPE_BUFFER, &aTensorBufferDesc };

        DML_BUFFER_TENSOR_DESC bTensorBufferDesc = {};
        bTensorBufferDesc.DataType = dmlDtype(dtype);
        bTensorBufferDesc.Flags = DML_TENSOR_FLAG_NONE;
        bTensorBufferDesc.DimensionCount = 4;
        bTensorBufferDesc.Sizes = bSizes;
        bTensorBufferDesc.TotalTensorSizeInBytes = bBytes;
        DML_TENSOR_DESC bTensorDesc = { DML_TENSOR_TYPE_BUFFER, &bTensorBufferDesc };

DML_BUFFER_TENSOR_DESC cTensorBufferDesc = {};
        cTensorBufferDesc.DataType = dmlDtype(dtype);
        cTensorBufferDesc.Flags = DML_TENSOR_FLAG_NONE;
        cTensorBufferDesc.DimensionCount = 4;
        cTensorBufferDesc.Sizes = cSizes;
        cTensorBufferDesc.TotalTensorSizeInBytes = cBytes;
        DML_TENSOR_DESC cTensorDesc = { DML_TENSOR_TYPE_BUFFER, &cTensorBufferDesc };

        DML_GEMM_OPERATOR_DESC gemmDesc = {};
        gemmDesc.ATensor = &aTensorDesc;
        gemmDesc.BTensor = &bTensorDesc;
        gemmDesc.CTensor = nullptr;
        gemmDesc.OutputTensor = &cTensorDesc;
        gemmDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
        gemmDesc.TransB = DML_MATRIX_TRANSFORM_NONE;
        gemmDesc.Alpha = 1.0f;
        gemmDesc.Beta = 0.0f;
        gemmDesc.FusedActivation = nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_GEMM, &gemmDesc };
        CachedOp cached = compileOperator(&opDesc);
        if (!cached.compiledOp) return -5;
        g_handleOpCache[shapeKey] = cached;
        it = g_handleOpCache.find(shapeKey);
    }

    CachedOp& op = it->second;

    DML_BUFFER_BINDING aBinding = { ta.resource, 0, aBytes };
    DML_BUFFER_BINDING bBinding = { tb.resource, 0, bBytes };
    DML_BUFFER_BINDING cBinding = { tc.resource, 0, cBytes };
    DML_BINDING_DESC inputDescs[] = {
        { DML_BINDING_TYPE_BUFFER, &aBinding },
        { DML_BINDING_TYPE_BUFFER, &bBinding },
        { DML_BINDING_TYPE_NONE, nullptr }
    };
    DML_BINDING_DESC outputDesc = { DML_BINDING_TYPE_BUFFER, &cBinding };
    dispatchOperator(op, inputDescs, 3, &outputDesc);

    g_gemmCount++;
    g_gemmDispatchMs += nowMs() - t0;
    return 0;
}

// ── Stage 1: 裝置端 elementwise multiply（SwiGLU 的 silu(x)*y 融合）──
int dml_elementwise_mul(uint64_t a, uint64_t b, uint64_t c, uint32_t count, uint32_t dtype) {
    if (!g_initialized) {
        if (dml_gemm_init() != 0) return -1;
    }
    if (a >= g_tensors.size() || b >= g_tensors.size() || c >= g_tensors.size()) return -3;
    TensorEntry& ta = g_tensors[a];
    TensorEntry& tb = g_tensors[b];
    TensorEntry& tc = g_tensors[c];
    if (!ta.in_use || !tb.in_use || !tc.in_use) return -3;

    size_t elemSize = dtypeSize(dtype);
    size_t bytes = (size_t)count * elemSize;
    if (ta.bytes < bytes || tb.bytes < bytes || tc.bytes < bytes) return -4;

    double t0 = nowMs();

    auto shapeKey = std::make_tuple(count, dtype);
    auto it = g_mulCache.find(shapeKey);
    if (it == g_mulCache.end()) {
        uint32_t sizes[] = { 1, 1, 1, count };

        DML_BUFFER_TENSOR_DESC aTensorBufferDesc = {};
        aTensorBufferDesc.DataType = dmlDtype(dtype);
        aTensorBufferDesc.Flags = DML_TENSOR_FLAG_NONE;
        aTensorBufferDesc.DimensionCount = 4;
        aTensorBufferDesc.Sizes = sizes;
        aTensorBufferDesc.TotalTensorSizeInBytes = bytes;
        DML_TENSOR_DESC aTensorDesc = { DML_TENSOR_TYPE_BUFFER, &aTensorBufferDesc };

        DML_BUFFER_TENSOR_DESC bTensorBufferDesc = aTensorBufferDesc;
        DML_TENSOR_DESC bTensorDesc = { DML_TENSOR_TYPE_BUFFER, &bTensorBufferDesc };

        DML_BUFFER_TENSOR_DESC cTensorBufferDesc = aTensorBufferDesc;
        DML_TENSOR_DESC cTensorDesc = { DML_TENSOR_TYPE_BUFFER, &cTensorBufferDesc };

        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mulDesc = {};
        mulDesc.ATensor = &aTensorDesc;
        mulDesc.BTensor = &bTensorDesc;
        mulDesc.OutputTensor = &cTensorDesc;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &mulDesc };
        CachedOp cached = compileOperator(&opDesc);
        if (!cached.compiledOp) return -5;
        g_mulCache[shapeKey] = cached;
        it = g_mulCache.find(shapeKey);
    }

    CachedOp& op = it->second;

    DML_BUFFER_BINDING aBinding = { ta.resource, 0, bytes };
    DML_BUFFER_BINDING bBinding = { tb.resource, 0, bytes };
    DML_BUFFER_BINDING cBinding = { tc.resource, 0, bytes };
    DML_BINDING_DESC inputDescs[] = {
        { DML_BINDING_TYPE_BUFFER, &aBinding },
        { DML_BINDING_TYPE_BUFFER, &bBinding }
    };
    DML_BINDING_DESC outputDesc = { DML_BINDING_TYPE_BUFFER, &cBinding };
    dispatchOperator(op, inputDescs, 2, &outputDesc);

    g_gemmCount++;
    g_gemmDispatchMs += nowMs() - t0;
    return 0;
}

// ── Stage 1: 裝置端 sigmoid（silu(x) = x * sigmoid(x) 分解用）──
int dml_activation_sigmoid(uint64_t a, uint64_t c, uint32_t count, uint32_t dtype) {
    if (!g_initialized) {
        if (dml_gemm_init() != 0) return -1;
    }
    if (a >= g_tensors.size() || c >= g_tensors.size()) return -3;
    TensorEntry& ta = g_tensors[a];
    TensorEntry& tc = g_tensors[c];
    if (!ta.in_use || !tc.in_use) return -3;

    size_t elemSize = dtypeSize(dtype);
    size_t bytes = (size_t)count * elemSize;
    if (ta.bytes < bytes || tc.bytes < bytes) return -4;

    double t0 = nowMs();

    auto shapeKey = std::make_tuple(count, dtype);
    auto it = g_sigmoidCache.find(shapeKey);
    if (it == g_sigmoidCache.end()) {
        uint32_t sizes[] = { 1, 1, 1, count };

        DML_BUFFER_TENSOR_DESC aTensorBufferDesc = {};
        aTensorBufferDesc.DataType = dmlDtype(dtype);
        aTensorBufferDesc.Flags = DML_TENSOR_FLAG_NONE;
        aTensorBufferDesc.DimensionCount = 4;
        aTensorBufferDesc.Sizes = sizes;
        aTensorBufferDesc.TotalTensorSizeInBytes = bytes;
        DML_TENSOR_DESC aTensorDesc = { DML_TENSOR_TYPE_BUFFER, &aTensorBufferDesc };

        DML_BUFFER_TENSOR_DESC cTensorBufferDesc = aTensorBufferDesc;
        DML_TENSOR_DESC cTensorDesc = { DML_TENSOR_TYPE_BUFFER, &cTensorBufferDesc };

        DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigDesc = {};
        sigDesc.InputTensor = &aTensorDesc;
        sigDesc.OutputTensor = &cTensorDesc;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ACTIVATION_SIGMOID, &sigDesc };
        CachedOp cached = compileOperator(&opDesc);
        if (!cached.compiledOp) return -5;
        g_sigmoidCache[shapeKey] = cached;
        it = g_sigmoidCache.find(shapeKey);
    }

    CachedOp& op = it->second;

    DML_BUFFER_BINDING aBinding = { ta.resource, 0, bytes };
    DML_BUFFER_BINDING cBinding = { tc.resource, 0, bytes };
    DML_BINDING_DESC inputDescs[] = {
        { DML_BINDING_TYPE_BUFFER, &aBinding }
    };
    DML_BINDING_DESC outputDesc = { DML_BINDING_TYPE_BUFFER, &cBinding };
    dispatchOperator(op, inputDescs, 1, &outputDesc);

    g_gemmCount++;
    g_gemmDispatchMs += nowMs() - t0;
    return 0;
}

void dml_gemm_shutdown() {
    // ── Stage 0: dump 剖析統計 ──
    if (g_gemmCount > 0) {
        fprintf(stderr,
                "[dml_gemm] stats: count=%llu | H2D=%.2fms | dispatch=%.2fms | D2H=%.2fms | fence=%.2fms | total=%.2fms\n",
                (unsigned long long)g_gemmCount, g_gemmH2DMs, g_gemmDispatchMs, g_gemmD2HMs, g_gemmFenceMs,
                g_gemmH2DMs + g_gemmDispatchMs + g_gemmD2HMs + g_gemmFenceMs);
    }

    for (auto& pair : g_opCache) {
        pair.second.bindingTable->Release();
        pair.second.descriptorHeap->Release();
        pair.second.compiledOp->Release();
    }
    g_opCache.clear();

    for (auto& pair : g_handleOpCache) {
        pair.second.bindingTable->Release();
        pair.second.descriptorHeap->Release();
        pair.second.compiledOp->Release();
    }
    g_handleOpCache.clear();

for (auto& pair : g_mulCache) {
        pair.second.bindingTable->Release();
        pair.second.descriptorHeap->Release();
        pair.second.compiledOp->Release();
    }
    g_mulCache.clear();

    for (auto& pair : g_sigmoidCache) {
        pair.second.bindingTable->Release();
        pair.second.descriptorHeap->Release();
        pair.second.compiledOp->Release();
    }
    g_sigmoidCache.clear();

    for (auto& t : g_tensors) {
        if (t.resource) t.resource->Release();
    }
    g_tensors.clear();
    g_freeTensorIds.clear();

    if (g_uploadA) g_uploadA->Release();
    if (g_inputA) g_inputA->Release();
    if (g_uploadB) g_uploadB->Release();
    if (g_inputB) g_inputB->Release();
    if (g_outputC) g_outputC->Release();
    if (g_readbackC) g_readbackC->Release();

    if (g_commandRecorder) g_commandRecorder->Release();
    if (g_dml_device) g_dml_device->Release();
    if (g_fence) g_fence->Release();
    if (g_commandList) g_commandList->Release();
    if (g_commandAllocator) g_commandAllocator->Release();
    if (g_commandQueue) g_commandQueue->Release();
    if (g_d3d_device) g_d3d_device->Release();
    if (g_adapter) g_adapter->Release();
    if (g_factory) g_factory->Release();
    g_initialized = false;
}

} // extern "C"
