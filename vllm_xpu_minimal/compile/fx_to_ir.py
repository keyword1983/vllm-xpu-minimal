"""fx graph → 自家 IR（Phase 8 Stage A2）。

支援的 fx 節點（Stage A 範圍，對應 Qwen2MLP / 標準 SwiGLU MLP）：
- placeholder       → ARG（依被 GEMM 引用與否判別 weight / input）
- call_function
  - linear.default   → GEMM（B 權重在轉換時轉置，IR 直接 A @ B）
  - silu.default     → SIGMOID + MUL（silu = x * sigmoid(x)，融合留 Stage B pass）
  - mul.Tensor       → MUL
  - slice.Tensor     → SLICE
- output            → OUT
"""
from __future__ import annotations

import torch
import torch.fx as fx

from vllm_xpu_minimal.compile.ir import IRGraph, IRNode

_OPS: dict[str, str] = {
    "linear.default": "GEMM",
    "mul.Tensor": "MUL",
}


def _is_weight_target(node: fx.Node) -> bool:
    """linear.default 的 args[1]（B 參數）即權重節點。"""
    if node.op != "placeholder":
        return False
    for user in node.users:
        if user.op == "call_function":
            target = getattr(user.target, "__name__", str(user.target))
            if target == "linear.default" and len(user.args) > 1:
                if user.args[1] is node:
                    return True
    return False


def fx_to_ir(ep) -> IRGraph:
    """torch.export 產物（ExportedProgram）→ IRGraph。

    權重處理：linear(x, w) 語意為 x @ w.T → 轉換時把 w 轉置存進
    graph.weights（IR 的 GEMM 直接 A @ B，無 transpose 語意）。
    """
    graph = IRGraph()

    for node in ep.graph.nodes:
        name = node.name

        if node.op == "placeholder":
            role = "weight" if _is_weight_target(node) else "input"
            if role == "weight":
                # 節點名統一使用 _t 後綴（與 GEMM 的 B 引用、bind_weights 寫入一致）
                graph.add_node(
                    IRNode(name + "_t", "ARG", attrs={"role": "weight"})
                )
            else:
                graph.add_node(
                    IRNode(
                        name,
                        "ARG",
                        attrs={"role": "input", "input_name": name},
                    )
                )
            continue

        if node.op == "call_function":
            target = getattr(node.target, "__name__", str(node.target))
            predecessors = [
                a.name for a in node.args if isinstance(a, fx.Node)
            ]

            if target == "linear.default":
                x_node, w_node = node.args[0], node.args[1]
                # B 權重：轉置後綁定（[N,K] → [K,N]，IR GEMM 直接 A@B）
                w_meta = w_node.meta.get("val")
                if w_meta is not None:
                    wt = w_meta.t().contiguous()
                else:
                    # meta 缺失時以 shapes 猜測（fallback，實務不應發生）
                    shapes = node.meta.get("val", torch.empty(0)).shape
                    wt = torch.empty((shapes[-1], w_meta.shape[0])) if w_meta is not None else None
                # 佔位 weight 節點：實際 tensor 由 bind_weights() 填入
                graph.weights[w_node.name + "_t"] = wt if wt is not None else w_meta
                graph.add_node(
                    IRNode(
                        name,
                        "GEMM",
                        inputs=[x_node.name, w_node.name + "_t"],
                    )
                )
                continue

            if target == "silu.default":
                x_node = node.args[0]
                sig_name = f"{name}__sig"
                graph.add_node(IRNode(sig_name, "SIGMOID", inputs=[x_node.name]))
                graph.add_node(IRNode(name, "MUL", inputs=[x_node.name, sig_name]))
                continue

            if target == "mul.Tensor":
                graph.add_node(
                    IRNode(name, "MUL", inputs=predecessors)
                )
                continue

            if target == "slice.Tensor":
                x_node = node.args[0]
                dim = int(node.args[1]) if len(node.args) > 1 else -1
                start = int(node.args[2]) if len(node.args) > 2 else None
                stop = int(node.args[3]) if len(node.args) > 3 else None
                step = int(node.args[4]) if len(node.args) > 4 else 1
                graph.add_node(
                    IRNode(
                        name,
                        "SLICE",
                        inputs=[x_node.name],
                        attrs={"dim": dim, "start": start, "stop": stop, "step": step},
                    )
                )
                continue

            raise NotImplementedError(
                f"fx op 未支援: {target}（節點 {name}）——需擴充 fx_to_ir"
            )

        if node.op == "output":
            out_arg = node.args[0]
            # torch.export 的 output args[0] 是 tuple（(node,)）包裝
            if isinstance(out_arg, (list, tuple)) and out_arg:
                out_arg = out_arg[0]
            out_name = out_arg.name if isinstance(out_arg, fx.Node) else None
            if out_name is not None:
                graph.output_name = f"{out_name}__out"
                graph.add_node(IRNode(graph.output_name, "OUT", inputs=[out_name]))
            continue

        raise NotImplementedError(f"fx 節點類型未支援: {node.op}（{name}）")

    return graph


def bind_weights(
    graph: IRGraph,
    param_names: dict[str, torch.Tensor],
) -> None:
    """把 fx placeholder 名 → 實際權重 tensor 填入 graph.weights（轉置後）。"""
    for fx_name, tensor in param_names.items():
        key = f"{fx_name}_t"
        if key in graph.weights:
            graph.weights[key] = tensor.detach().t().contiguous()


def export_mlp_to_ir(module: torch.nn.Module, dummy_input: torch.Tensor) -> IRGraph:
    """一鍵流程：torch.export → fx_to_ir → bind_weights。

    回傳的 IRGraph 可直接用 interpret() 對拍 reference。
    """
    from torch.export import export

    ep = export(module, (dummy_input,))
    graph = fx_to_ir(ep)

    # 收集 placeholder → 實際參數（module 的 state_dict 對應 fx 參數名）
    # fx 參數名形如 p_gate_up_proj_weight；對應 module 的 gate_up_proj.weight
    param_map: dict[str, torch.Tensor] = {}
    for node in ep.graph.nodes:
        if node.op == "placeholder":
            # 從 graph 的 lifted 參數名還原 module 路徑（fx 用 _param 前綴）
            raw = node.name
            # torch.export 會 lift 參數為 placeholder（名字: p_<module_path>_weight 等）
            if raw.startswith("p_"):
                # p_gate_up_proj_weight → gate_up_proj.weight
                parts = raw[2:].split("_")
                # 保守做法：用 meta 對應的 module 路徑
                if node.meta and "example_value" in node.meta:
                    pass
            # 直接以名字比對 module 的 named_parameters
            for pname, ptensor in module.named_parameters():
                fx_key = "p_" + pname.replace(".", "_")
                if fx_key == raw or (raw.endswith("_weight") and pname.replace(".", "_") in raw):
                    param_map[raw] = ptensor
                    break

    bind_weights(graph, param_map)
    return graph