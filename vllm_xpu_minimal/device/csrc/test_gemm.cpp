#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstring>
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

void CloseExecuteResetWait(
    ID3D12CommandQueue* commandQueue,
    ID3D12CommandAllocator* commandAllocator,
    ID3D12GraphicsCommandList* commandList,
    ID3D12Fence* fence,
    uint64_t& fenceValue)
{
    HRESULT hr = commandList->Close();
    if (FAILED(hr)) std::cerr << "commandList->Close failed: 0x" << std::hex << hr << std::endl;
    ID3D12CommandList* lists[] = { commandList };
    commandQueue->ExecuteCommandLists(1, lists);

    fenceValue++;
    hr = commandQueue->Signal(fence, fenceValue);
    if (FAILED(hr)) std::cerr << "commandQueue->Signal failed: 0x" << std::hex << hr << std::endl;
    while (fence->GetCompletedValue() < fenceValue) {
        std::this_thread::yield();
    }
    hr = commandAllocator->Reset();
    if (FAILED(hr)) std::cerr << "commandAllocator->Reset failed: 0x" << std::hex << hr << std::endl;
    hr = commandList->Reset(commandAllocator, nullptr);
    if (FAILED(hr)) std::cerr << "commandList->Reset failed: 0x" << std::hex << hr << std::endl;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "🚀 [DirectML Hardware GEMM] Official Pipeline on Intel UHD 630" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. 動態連結系統動態庫
    void* dxcore_lib = dlopen("/usr/lib/wsl/lib/libdxcore.so", RTLD_NOW);
    void* d3d12_lib  = dlopen("/usr/lib/wsl/lib/libd3d12.so", RTLD_NOW);
    void* dml_lib    = dlopen("/mnt/d/work/xpu/vllm-xpu-minimal/vllm_xpu_minimal/device/lib/libdirectml.so", RTLD_NOW);

    if (!dxcore_lib || !d3d12_lib || !dml_lib) {
        std::cerr << "Failed to dlopen essential libraries!" << std::endl;
        return 1;
    }

    auto DXCoreCreateAdapterFactory_ptr = (DXCoreCreateAdapterFactoryFn)dlsym(dxcore_lib, "DXCoreCreateAdapterFactory");
    auto D3D12CreateDevice_ptr = (D3D12CreateDeviceFn)dlsym(d3d12_lib, "D3D12CreateDevice");
    auto DMLCreateDevice_ptr = (DMLCreateDeviceFn)dlsym(dml_lib, "DMLCreateDevice");

    // 2. 透過 DXCore 取得 Intel UHD 630 適配器
    IDXCoreAdapterFactory* factory = nullptr;
    DXCoreCreateAdapterFactory_ptr(IID_IDXCoreAdapterFactory_val, (void**)&factory);

    static const GUID DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE_val = { 0x248e2800, 0xa793, 0x4724, { 0xab, 0xaa, 0x23, 0xa6, 0xde, 0x1b, 0xe0, 0x90 } };
    IDXCoreAdapterList* adapterList = nullptr;
    const GUID filterGuids[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE_val };
    factory->CreateAdapterList(1, filterGuids, IID_IDXCoreAdapterList_val, (void**)&adapterList);

    IDXCoreAdapter* adapter = nullptr;
    adapterList->GetAdapter(0, IID_IDXCoreAdapter_val, (void**)&adapter);

    size_t descSize = 0;
    adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &descSize);
    char* devName = new char[descSize];
    adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, descSize, devName);
    std::cout << "✓ 鎖定物理計算卡: " << devName << std::endl;
    delete[] devName;

    // 3. 建立 D3D12 裝置與指令管線
    ID3D12Device* d3d_device = nullptr;
    D3D12CreateDevice_ptr(adapter, D3D_FEATURE_LEVEL_11_0, IID_ID3D12Device_val, (void**)&d3d_device);

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* commandQueue = nullptr;
    d3d_device->CreateCommandQueue(&queueDesc, IID_ID3D12CommandQueue_val, (void**)&commandQueue);

    ID3D12CommandAllocator* commandAllocator = nullptr;
    d3d_device->CreateCommandAllocator(queueDesc.Type, IID_ID3D12CommandAllocator_val, (void**)&commandAllocator);

    ID3D12GraphicsCommandList* commandList = nullptr;
    d3d_device->CreateCommandList(0, queueDesc.Type, commandAllocator, nullptr, IID_ID3D12GraphicsCommandList_val, (void**)&commandList);

    ID3D12Fence* fence = nullptr;
    d3d_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence_val, (void**)&fence);
    uint64_t fenceValue = 0;

    // 4. 建立 DirectML 裝置與指令記錄器
    IDMLDevice* dml_device = nullptr;
    DMLCreateDevice_ptr(d3d_device, DML_CREATE_DEVICE_FLAG_NONE, IID_IDMLDevice_val, (void**)&dml_device);

    IDMLCommandRecorder* commandRecorder = nullptr;
    dml_device->CreateCommandRecorder(IID_IDMLCommandRecorder_val, (void**)&commandRecorder);

    // 5. 定義 GEMM 矩陣形狀: A (M x K) * B (K x N) = C (M x N)
    const uint32_t M = 512;
    const uint32_t K = 512;
    const uint32_t N = 512;
    std::cout << "✓ 設定測試 GEMM 維度: A(" << M << "x" << K << ") * B(" << K << "x" << N << ") = C(" << M << "x" << N << ")" << std::endl;

    size_t aBufferSize = (size_t)M * K * sizeof(float);
    size_t bBufferSize = (size_t)K * N * sizeof(float);
    size_t cBufferSize = (size_t)M * N * sizeof(float);

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
    dml_device->CreateOperator(&opDesc, IID_IDMLOperator_val, (void**)&dmlOperator);

    IDMLCompiledOperator* compiledOperator = nullptr;
    dml_device->CompileOperator(dmlOperator, DML_EXECUTION_FLAG_NONE, IID_IDMLCompiledOperator_val, (void**)&compiledOperator);
    std::cout << "✓ DirectML GEMM 算子編譯完成" << std::endl;

    // 6. 建立 Operator Initializer (官方標準必經步驟)
    IDMLOperatorInitializer* operatorInitializer = nullptr;
    IDMLCompiledOperator* compiledOperators[] = { compiledOperator };
    dml_device->CreateOperatorInitializer(1, compiledOperators, IID_IDMLOperatorInitializer_val, (void**)&operatorInitializer);

    DML_BINDING_PROPERTIES initProps = operatorInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES execProps = compiledOperator->GetBindingProperties();
    uint32_t descriptorCount = std::max(initProps.RequiredDescriptorCount, execProps.RequiredDescriptorCount);
    std::cout << "✓ 描述符數量: init=" << initProps.RequiredDescriptorCount 
              << ", exec=" << execProps.RequiredDescriptorCount 
              << ", max=" << descriptorCount << std::endl;

    // 建立 Descriptor Heap
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = descriptorCount;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    ID3D12DescriptorHeap* descriptorHeap = nullptr;
    if (descriptorCount > 0) {
        d3d_device->CreateDescriptorHeap(&heapDesc, IID_ID3D12DescriptorHeap_val, (void**)&descriptorHeap);
        ID3D12DescriptorHeap* heaps[] = { descriptorHeap };
        commandList->SetDescriptorHeaps(1, heaps);
    }

    // 建立 Binding Table
    DML_BINDING_TABLE_DESC tableDesc = {};
    tableDesc.Dispatchable = operatorInitializer;
    tableDesc.CPUDescriptorHandle = descriptorHeap ? descriptorHeap->GetCPUDescriptorHandleForHeapStart() : D3D12_CPU_DESCRIPTOR_HANDLE{};
    tableDesc.GPUDescriptorHandle = descriptorHeap ? descriptorHeap->GetGPUDescriptorHandleForHeapStart() : D3D12_GPU_DESCRIPTOR_HANDLE{};
    tableDesc.SizeInDescriptors = descriptorCount;

    IDMLBindingTable* bindingTable = nullptr;
    dml_device->CreateBindingTable(&tableDesc, IID_IDMLBindingTable_val, (void**)&bindingTable);

    // 執行算子初始化 (Dispatch Initializer)
    if (initProps.RequiredDescriptorCount > 0 || initProps.PersistentResourceSize > 0) {
        commandRecorder->RecordDispatch(commandList, operatorInitializer, bindingTable);
        CloseExecuteResetWait(commandQueue, commandAllocator, commandList, fence, fenceValue);
        std::cout << "✓ 算子初始化派發完成" << std::endl;
    }

    // 7. 配置 DEFAULT 堆與 UPLOAD 堆 (微軟標準架構: 計算必在 DEFAULT 顯存執行)
    D3D12_HEAP_PROPERTIES defaultHeap = { D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    D3D12_HEAP_PROPERTIES uploadHeap  = { D3D12_HEAP_TYPE_UPLOAD,  D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    D3D12_HEAP_PROPERTIES readbackHeap= { D3D12_HEAP_TYPE_READBACK,D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };

    auto makeBufferDesc = [](size_t size, D3D12_RESOURCE_FLAGS flags) {
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
    };

    // GPU 板載/共享顯存計算張量 (DEFAULT 堆，支援 UNORDERED_ACCESS)
    ID3D12Resource* inputBufferA = nullptr;
    ID3D12Resource* inputBufferB = nullptr;
    ID3D12Resource* outputBufferC = nullptr;

    auto descDefaultA = makeBufferDesc(aBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto descDefaultB = makeBufferDesc(bBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto descDefaultC = makeBufferDesc(cBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    d3d_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &descDefaultA, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_ID3D12Resource_val, (void**)&inputBufferA);
    d3d_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &descDefaultB, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_ID3D12Resource_val, (void**)&inputBufferB);
    d3d_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &descDefaultC, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_ID3D12Resource_val, (void**)&outputBufferC);

    // CPU 上傳與讀回緩衝區
    ID3D12Resource* uploadBufferA = nullptr;
    ID3D12Resource* uploadBufferB = nullptr;
    ID3D12Resource* readbackBufferC = nullptr;

    auto descUploadA = makeBufferDesc(aBufferSize, D3D12_RESOURCE_FLAG_NONE);
    auto descUploadB = makeBufferDesc(bBufferSize, D3D12_RESOURCE_FLAG_NONE);
    auto descReadback = makeBufferDesc(cBufferSize, D3D12_RESOURCE_FLAG_NONE);

    d3d_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &descUploadA, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_ID3D12Resource_val, (void**)&uploadBufferA);
    d3d_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &descUploadB, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_ID3D12Resource_val, (void**)&uploadBufferB);
    d3d_device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &descReadback, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_ID3D12Resource_val, (void**)&readbackBufferC);

    // 8. 準備輸入數據並寫入 UPLOAD 堆 (A=1.0, B=2.0)
    std::vector<float> aHost(M * K, 1.0f);
    std::vector<float> bHost(K * N, 2.0f);

    float* aMapped = nullptr;
    uploadBufferA->Map(0, nullptr, (void**)&aMapped);
    memcpy(aMapped, aHost.data(), aBufferSize);
    uploadBufferA->Unmap(0, nullptr);

    float* bMapped = nullptr;
    uploadBufferB->Map(0, nullptr, (void**)&bMapped);
    memcpy(bMapped, bHost.data(), bBufferSize);
    uploadBufferB->Unmap(0, nullptr);

    // 9. H2D 拷貝至 DEFAULT 顯存，並切換狀態至 UNORDERED_ACCESS
    commandList->CopyBufferRegion(inputBufferA, 0, uploadBufferA, 0, aBufferSize);
    commandList->CopyBufferRegion(inputBufferB, 0, uploadBufferB, 0, bBufferSize);

    D3D12_RESOURCE_BARRIER uploadBarriers[2] = {};
    uploadBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    uploadBarriers[0].Transition.pResource = inputBufferA;
    uploadBarriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    uploadBarriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    uploadBarriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    uploadBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    uploadBarriers[1].Transition.pResource = inputBufferB;
    uploadBarriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    uploadBarriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    uploadBarriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    commandList->ResourceBarrier(2, uploadBarriers);

    CloseExecuteResetWait(commandQueue, commandAllocator, commandList, fence, fenceValue);
    std::cout << "✓ H2D 顯存數據傳輸與狀態切換完畢 (UNORDERED_ACCESS 就緒)" << std::endl;

    // 10. 重置 Binding Table 綁定 GEMM 執行算子
    if (descriptorHeap) {
        ID3D12DescriptorHeap* heaps[] = { descriptorHeap };
        commandList->SetDescriptorHeaps(1, heaps);
    }

    tableDesc.Dispatchable = compiledOperator;
    HRESULT hr = bindingTable->Reset(&tableDesc);
    if (FAILED(hr)) {
        std::cerr << "bindingTable->Reset failed: 0x" << std::hex << hr << std::endl;
        return 1;
    }
    std::cout << "✓ bindingTable->Reset 成功" << std::endl;

    DML_BUFFER_BINDING aBinding = { inputBufferA, 0, aBufferSize };
    DML_BUFFER_BINDING bBinding = { inputBufferB, 0, bBufferSize };
    DML_BUFFER_BINDING cBinding = { outputBufferC, 0, cBufferSize };

    DML_BINDING_DESC inputBindings[] = {
        { DML_BINDING_TYPE_BUFFER, &aBinding },
        { DML_BINDING_TYPE_BUFFER, &bBinding },
        { DML_BINDING_TYPE_NONE, nullptr }
    };
    bindingTable->BindInputs(3, inputBindings);

    DML_BINDING_DESC outputBinding = { DML_BINDING_TYPE_BUFFER, &cBinding };
    bindingTable->BindOutputs(1, &outputBinding);

    // 11. 記錄執行命令並派發 GPU
    std::cout << "🔥 正在發射 GEMM 指令隊列至 Intel UHD 630 GPU 核心..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    commandRecorder->RecordDispatch(commandList, compiledOperator, bindingTable);
    CloseExecuteResetWait(commandQueue, commandAllocator, commandList, fence, fenceValue);

    auto end_time = std::chrono::high_resolution_clock::now();
    double exec_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "✓ Intel UHD 630 GPU 運算完畢！執行耗時: " << exec_ms << " ms" << std::endl;
    std::cout << "✓ D3D12 Device Removed Reason: 0x" << std::hex << d3d_device->GetDeviceRemovedReason() << std::endl;

    // 12. D2H 讀回: Transition outputBufferC -> COPY_SOURCE, 複製到 readbackBufferC
    D3D12_RESOURCE_BARRIER readbackBarrier = {};
    readbackBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    readbackBarrier.Transition.pResource = outputBufferC;
    readbackBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    readbackBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    readbackBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    commandList->ResourceBarrier(1, &readbackBarrier);

    commandList->CopyResource(readbackBufferC, outputBufferC);
    CloseExecuteResetWait(commandQueue, commandAllocator, commandList, fence, fenceValue);

    // 13. 讀回數值校驗
    D3D12_RANGE readRange = { 0, cBufferSize };
    float* cMapped = nullptr;
    readbackBufferC->Map(0, &readRange, (void**)&cMapped);

    float expected = 1.0f * 2.0f * (float)K; // 1024.0
    float actualSample = cMapped[0];
    float maxDiff = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        float diff = std::abs(cMapped[i] - expected);
        if (diff > maxDiff) maxDiff = diff;
    }
    readbackBufferC->Unmap(0, nullptr);

    std::cout << "✓ 採樣輸出數值: " << actualSample << " (期望值: " << expected << ")" << std::endl;
    std::cout << "✓ 最大絕對誤差 (Max Diff): " << maxDiff << std::endl;

    if (maxDiff < 1e-3) {
        std::cout << "==========================================================" << std::endl;
        std::cout << "🎉【驗證 100% 成功】Intel UHD 630 實體顯卡 GEMM 運算數值精確無誤！" << std::endl;
        std::cout << "==========================================================" << std::endl;
    } else {
        std::cerr << "⚠ 數值比對誤差過大！" << std::endl;
        return 1;
    }

    // 釋放資源
    fence->Release();
    readbackBufferC->Release();
    outputBufferC->Release();
    uploadBufferB->Release();
    uploadBufferA->Release();
    inputBufferB->Release();
    inputBufferA->Release();
    if (descriptorHeap) descriptorHeap->Release();
    bindingTable->Release();
    commandRecorder->Release();
    operatorInitializer->Release();
    compiledOperator->Release();
    dmlOperator->Release();
    dml_device->Release();
    commandList->Release();
    commandAllocator->Release();
    commandQueue->Release();
    d3d_device->Release();
    adapter->Release();
    adapterList->Release();
    factory->Release();
    return 0;
}
