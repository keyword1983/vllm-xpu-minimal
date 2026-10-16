"""自家 IR 資料結構與直譯執行器（Phase 8 Stage A/B）。

IR 設計原則：JSON-serializable（之後 codegen 給 C++ 圖執行器吃）、
保留高層 op 語意（GEMM/MUL/SIGMOID/SLICE…）、權重與圖輸入分離。

【op 集（Stage A 範圍）】
- ARG     圖輸入/權重佔位（role: input|weight）
- GEMM    A @ B（B 來自 weights dict，fx_to_ir 時已處理轉置）
- SIGMOID sigmoid(x)
- MUL     elementwise a * b
- SLICE   x[..., start:end:step]（dim=-1 等）
- OUT     圖輸出標記
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

# ---- IR 節點 ----


@dataclass
class IRNode:
    name: str
    op: str
    inputs: list[str] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


# ---- IR 圖 ----


class IRGraph:
    """節點表 + 權重表。節點為拓樸序（fx graph 順序即拓樸序）。"""

    def __init__(self) -> None:
        self.nodes: dict[str, IRNode] = {}
        self.output_name: str | None = None
        # 權重綁定：name → tensor（GEMM 的 B 等，不隨圖輸入變動）
        self.weights: dict[str, torch.Tensor] = {}
        self.order: list[str] = []

    def add_node(self, node: IRNode) -> None:
        self.nodes[node.name] = node
        self.order.append(node.name)

    def to_dict(self) -> dict:
        """JSON 友好序列化（權重只記錄 meta，tensor 另行傳遞）。"""
        return {
            "nodes": [
                {
                    "name": n.name,
                    "op": n.op,
                    "inputs": n.inputs,
                    "attrs": {
                        k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in n.attrs.items()
                        if k != "shape"
                    },
                }
                for n in self.nodes.values()
            ],
            "output": self.output_name,
            "order": self.order,
            "weights": {
                name: {
                    "shape": list(w.shape),
                    "dtype": str(w.dtype),
                    "data_ptr": w.data_ptr(),
                }
                for name, w in self.weights.items()
            },
        }

    def __repr__(self) -> str:
        return f"IRGraph(nodes={len(self.nodes)}, weights={list(self.weights)})"


# ---- 直譯執行器（A3 驗證用：與 CPU reference 對拍） ----


def interpret(graph: IRGraph, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """拓樸序執行 IR，回傳輸出張量。

    權重節點（ARG role=weight）從 graph.weights 取值；
    輸入節點（ARG role=input）從 inputs 取值。
    """
    values: dict[str, torch.Tensor] = {}

    for name in graph.order:
        node = graph.nodes[name]
        op = node.op
        if op == "ARG":
            role = node.attrs.get("role", "input")
            if role == "weight":
                values[name] = graph.weights[name]
            else:
                values[name] = inputs[node.attrs["input_name"]]
        elif op == "GEMM":
            a = values[node.inputs[0]]
            b = graph.weights[node.inputs[1]]
            values[name] = a @ b
        elif op == "SIGMOID":
            values[name] = torch.sigmoid(values[node.inputs[0]])
        elif op == "MUL":
            values[name] = values[node.inputs[0]] * values[node.inputs[1]]
        elif op == "SLICE":
            x = values[node.inputs[0]]
            dim = node.attrs["dim"]
            start = node.attrs.get("start")
            stop = node.attrs.get("stop")
            step = node.attrs.get("step", 1)
            idx = [slice(None)] * x.dim()
            idx[dim] = slice(start, stop, step)
            values[name] = x[tuple(idx)]
        elif op == "OUT":
            return values[node.inputs[0]]
        else:
            raise NotImplementedError(f"IR op 未支援: {op}")

    raise RuntimeError(f"IR 圖缺少 OUT 節點（output={graph.output_name}）")