#include <iostream>
#include <dlfcn.h>
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include <directx/dxcore.h>
#include <directx/DirectML.h>

// 官方精準 GUID 定義
static const GUID IID_IDXCoreAdapterFactory_val = { 0x78ee5945, 0xc36e, 0x4b13, { 0xa6, 0x69, 0x00, 0x5d, 0xd1, 0x1c, 0x0f, 0x06 } };
static const GUID IID_IDXCoreAdapterList_val    = { 0x526c7776, 0x40e9, 0x459b, { 0xb7, 0x11, 0xf3, 0x2a, 0xd7, 0x6d, 0xfc, 0x28 } };
static const GUID IID_IDXCoreAdapter_val        = { 0xf0db4c7f, 0xfe5a, 0x42a2, { 0xbd, 0x62, 0xf2, 0xa6, 0xcf, 0x6f, 0xc8, 0x3e } };
static const GUID DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS_val = { 0x0c9ece4d, 0x2f6e, 0x4f01, { 0x8c, 0x96, 0xe8, 0x9e, 0x33, 0x1b, 0x47, 0xb1 } };
static const GUID IID_ID3D12Device_val          = { 0x189819f1, 0x1db6, 0x4b57, { 0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7 } };
static const GUID IID_IDMLDevice_val            = { 0x6dbd6437, 0x96fd, 0x423f, { 0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f } };

typedef HRESULT (*DXCoreCreateAdapterFactoryFn)(REFIID riid, void** ppv);
typedef HRESULT (*D3D12CreateDeviceFn)(IUnknown* pAdapter, D3D_FEATURE_LEVEL MinimumFeatureLevel, REFIID riid, void** ppv);
typedef HRESULT (*DMLCreateDeviceFn)(ID3D12Device* d3d12Device, DML_CREATE_DEVICE_FLAGS flags, REFIID riid, void** ppv);

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "[DML Native Init] Testing DirectML & D3D12 via DXCore" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. 動態載入 libdxcore.so
    void* dxcore_lib = dlopen("/usr/lib/wsl/lib/libdxcore.so", RTLD_NOW);
    if (!dxcore_lib) {
        std::cerr << "Failed to dlopen libdxcore.so: " << dlerror() << std::endl;
        return 1;
    }
    auto DXCoreCreateAdapterFactory_ptr = (DXCoreCreateAdapterFactoryFn)dlsym(dxcore_lib, "DXCoreCreateAdapterFactory");

    // 2. 動態載入 libd3d12.so
    void* d3d12_lib = dlopen("/usr/lib/wsl/lib/libd3d12.so", RTLD_NOW);
    if (!d3d12_lib) {
        std::cerr << "Failed to dlopen libd3d12.so: " << dlerror() << std::endl;
        return 1;
    }
    auto D3D12CreateDevice_ptr = (D3D12CreateDeviceFn)dlsym(d3d12_lib, "D3D12CreateDevice");

    // 3. 動態載入 libdirectml.so
    void* dml_lib = dlopen("/mnt/d/work/xpu/vllm-xpu-minimal/vllm_xpu_minimal/device/lib/libdirectml.so", RTLD_NOW);
    if (!dml_lib) {
        std::cerr << "Failed to dlopen libdirectml.so: " << dlerror() << std::endl;
        return 1;
    }
    auto DMLCreateDevice_ptr = (DMLCreateDeviceFn)dlsym(dml_lib, "DMLCreateDevice");

    // 4. 使用 DXCore 枚舉適配器
    IDXCoreAdapterFactory* factory = nullptr;
    HRESULT hr = DXCoreCreateAdapterFactory_ptr(IID_IDXCoreAdapterFactory_val, (void**)&factory);
    if (FAILED(hr) || !factory) {
        std::cerr << "DXCoreCreateAdapterFactory failed: 0x" << std::hex << hr << std::endl;
        return 1;
    }

    // WSL2 下 GPU 是 D3D12_CORE_COMPUTE (純計算)
    static const GUID DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE_val = { 0x248e2800, 0xa793, 0x4724, { 0xab, 0xaa, 0x23, 0xa6, 0xde, 0x1b, 0xe0, 0x90 } };
    IDXCoreAdapterList* adapterList = nullptr;
    const GUID filterGuids[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE_val };
    hr = factory->CreateAdapterList(1, filterGuids, IID_IDXCoreAdapterList_val, (void**)&adapterList);
    if (FAILED(hr) || !adapterList) {
        std::cerr << "CreateAdapterList failed: 0x" << std::hex << hr << std::endl;
        return 1;
    }

    uint32_t adapterCount = adapterList->GetAdapterCount();
    std::cout << "✓ 偵測到的 DXCore GPU 適配器數量: " << adapterCount << std::endl;
    if (adapterCount == 0) {
        std::cerr << "未找到任何相容的 D3D12 GPU 適配器！" << std::endl;
        return 1;
    }

    IDXCoreAdapter* chosenAdapter = nullptr;
    hr = adapterList->GetAdapter(0, IID_IDXCoreAdapter_val, (void**)&chosenAdapter);
    if (FAILED(hr) || !chosenAdapter) {
        std::cerr << "GetAdapter failed: 0x" << std::hex << hr << std::endl;
        return 1;
    }

    // 查詢適配器描述 (名稱與顯存)
    size_t descSize = 0;
    if (SUCCEEDED(chosenAdapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &descSize))) {
        char* desc = new char[descSize];
        if (SUCCEEDED(chosenAdapter->GetProperty(DXCoreAdapterProperty::DriverDescription, descSize, desc))) {
            std::cout << "✓ 選定 GPU 裝置: " << desc << std::endl;
        }
        delete[] desc;
    }

    uint64_t dedicatedMem = 0;
    if (SUCCEEDED(chosenAdapter->GetProperty(DXCoreAdapterProperty::DedicatedAdapterMemory, sizeof(dedicatedMem), &dedicatedMem))) {
        std::cout << "  - 獨立顯存: " << (dedicatedMem / (1024 * 1024)) << " MB" << std::endl;
    }
    uint64_t sharedMem = 0;
    if (SUCCEEDED(chosenAdapter->GetProperty(DXCoreAdapterProperty::SharedSystemMemory, sizeof(sharedMem), &sharedMem))) {
        std::cout << "  - 共享系統記憶體: " << (sharedMem / (1024 * 1024)) << " MB" << std::endl;
    }

    // 5. 建立 D3D12 裝置
    ID3D12Device* d3d_device = nullptr;
    hr = D3D12CreateDevice_ptr(chosenAdapter, D3D_FEATURE_LEVEL_11_0, IID_ID3D12Device_val, (void**)&d3d_device);
    if (FAILED(hr) || !d3d_device) {
        std::cerr << "D3D12CreateDevice with adapter failed: 0x" << std::hex << hr << std::endl;
        return 1;
    }
    std::cout << "✓ 成功建立 ID3D12Device！" << std::endl;

    // 6. 建立 DirectML 裝置
    IDMLDevice* dml_device = nullptr;
    hr = DMLCreateDevice_ptr(d3d_device, DML_CREATE_DEVICE_FLAG_NONE, IID_IDMLDevice_val, (void**)&dml_device);
    if (FAILED(hr) || !dml_device) {
        std::cerr << "DMLCreateDevice failed: 0x" << std::hex << hr << std::endl;
        return 1;
    }
    std::cout << "✓ 成功建立 IDMLDevice (微軟原廠 DirectML 硬體核心接入成功)！" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "🎉 WSL2 直通 Intel UHD 630 的 DirectML 初始化 100% 成功！" << std::endl;
    std::cout << "========================================" << std::endl;

    dml_device->Release();
    d3d_device->Release();
    chosenAdapter->Release();
    adapterList->Release();
    factory->Release();
    return 0;
}
