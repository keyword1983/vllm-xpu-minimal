import os
import time

os.environ["VLLM_XPU_FAKE"] = "1"

from vllm import LLM, SamplingParams


def main():
    print("==========================================================")
    print("🚀 [End-to-End LLM] Initializing Qwen2.5-0.5B on Intel UHD 630 iGPU...")
    print("==========================================================")

    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=512,
        dtype="auto",
        enforce_eager=True,
    )

    prompt = "台灣的最高峰是哪一座山？請用繁體中文簡短回答。"
    sampling_params = SamplingParams(temperature=0.2, max_tokens=32)

    print("\n🔥 發送提示詞 (Prompt):", prompt)
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    t1 = time.perf_counter()

    print("\n==========================================================")
    for output in outputs:
        print(f"🎉 生成結果 (Generated): {output.outputs[0].text.strip()}")
    print(f"⏱️ 端到端推論耗時: {(t1 - t0):.2f} 秒")
    print("==========================================================")


if __name__ == "__main__":
    main()
