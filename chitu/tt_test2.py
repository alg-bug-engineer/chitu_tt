import os
import sys
import time
import statistics
from types import SimpleNamespace
import torch

# 确保路径包含 chitu 源码
sys.path.append("/workspace/chitu_cc")

from chitu.chitu_main import chitu_init, chitu_start, chitu_run, chitu_terminate
from chitu.backend import Backend
from chitu.task import Task, TaskPool, UserRequest
from chitu.task_type import TaskDecodeType

# --- 配置构建函数 ---
def build_args_tt_qwen_batch(hf_model_path: str | None = None):
    models = SimpleNamespace(
        name="TT-Qwen2.5",
        type="tt-qwen",
        ckpt_dir=hf_model_path,
        tokenizer_path=hf_model_path,
        tokenizer_type="hf",
    )
    infer = SimpleNamespace(
        seed=42,
        dp_size=1,
        tp_size=1,
        pp_size=1,
        ep_size=1,
        # 关键设置：max_reqs 必须 >= 32 以满足 TT 底层 Tile 对齐要求
        # 即使我们只测 batch=8，底层也需要分配 32 的空间
        max_reqs=32,  
        max_seq_len=1024, # 增加长度以适应多轮测试
        prefill_chunk_size=None,
        bind_process_to_cpu="none",
        use_cuda_graph="auto",
        attn_type="auto",
        op_impl="tt",
        mla_absorb="none",
        cache_type="skew",
        num_blocks=1,
        memory_utilization=0.98,
        raise_lower_bit_float_to="bfloat16",
    )
    dp_router = SimpleNamespace(is_router=False, pd_disaggregation=SimpleNamespace(enabled=False))
    dp_config = SimpleNamespace(dp_id=0, router=dp_router, scheduler_base_port=5557, scheduler_base_host="127.0.0.1", router_host="127.0.0.1", router_port=5556)
    
    # 使用 FCFS 简单调度
    scheduler = SimpleNamespace(type="fcfs")
    
    pp_config = SimpleNamespace(
        prefill_num_tasks=1,
        decode_num_tasks=1,
        prefill_num_tasks_divided_by_pp=False,
        enforce_decode_num_tasks_max=False,
    )
    
    args = SimpleNamespace(
        infer=infer,
        models=models,
        dp_config=dp_config,
        scheduler=scheduler,
        pp_config=pp_config,
        float_16bit_variant="bfloat16",
        skip_preprocess=False,
        debug=SimpleNamespace(skip_model_load=False),
    )
    return args

# --- 基准测试函数 ---
def run_benchmark(batch_size, max_new_tokens=64):
    # 准备测试 Prompt 池
    base_prompts = [
        "介绍下人工智能",
        "写一首关于春天的七言绝句",
        "Explain Quantum Computing simply",
        "1+1等于几？",
        "北京是哪个国家的首都？",
        "What is the capital of France?",
        "列举3个健康的水果",
        "Python中列表和元组的区别？"
    ]
    # 如果 batch_size 大于预设 prompt 数量，循环填充
    current_prompts = []
    while len(current_prompts) < batch_size:
        current_prompts.extend(base_prompts)
    current_prompts = current_prompts[:batch_size]
    
    print(f"\n" + "-"*60)
    print(f"🚀 开始测试 Batch Size = {batch_size}")
    print(f"-"*60)
    
    tasks = []
    # 构造任务
    for i, prompt in enumerate(current_prompts):
        req = UserRequest(
            message=[{"role": "user", "content": prompt}],
            request_id=f"bench-b{batch_size}-{i}",
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=1.0,
            top_k=1,
        )
        task = Task(task_id=req.request_id, req=req, stop_with_eos=True)
        TaskPool.add(task)
        tasks.append(task)

    # 统计变量
    start_time = time.perf_counter()
    first_token_times = [None] * batch_size
    completion_times = [None] * batch_size
    
    step = 0
    active_tasks = batch_size
    
    # 推理循环
    while active_tasks > 0:
        chitu_run()
        step += 1
        
        current_time = time.perf_counter()
        
        # 检查每个任务的状态
        completed_in_this_step = 0
        for i, task in enumerate(tasks):
            # 记录首字时间 (TTFT)
            if first_token_times[i] is None and task.num_new_tokens > 0:
                first_token_times[i] = current_time - start_time
            
            # 检查是否完成
            if completion_times[i] is None:
                is_stopped = task._decode_status in (TaskDecodeType.Stopped, TaskDecodeType.StopEOS, TaskDecodeType.StopLength)
                is_completed = task.req.completed.is_set()
                
                if is_stopped or is_completed:
                    completion_times[i] = current_time - start_time
                    # 打印部分生成结果用于验证
                    if hasattr(task, 'response') and len(task.response) > 0:
                        # 简略打印前10个token id证明在工作
                        token_preview = task.response.to_tensor().cpu().tolist()[:5]
                        print(f"  [Task {i}] 完成. Tokens: {task.num_new_tokens} Preview: {token_preview}...")
                    
        # 更新剩余任务数
        active_tasks = batch_size - sum(1 for t in completion_times if t is not None)
        
        # 超时保护
        if step > max_new_tokens * 2 + 50:
            print("⚠️ 警告: 达到最大步数限制，强制停止")
            break

    total_time = time.perf_counter() - start_time
    
    # --- 计算指标 ---
    total_tokens = sum(t.num_new_tokens for t in tasks)
    
    # 过滤掉未完成的任务（如果有的话）
    valid_completion_times = [t for t in completion_times if t is not None]
    valid_ttft_times = [t for t in first_token_times if t is not None]
    
    avg_latency = statistics.mean(valid_completion_times) if valid_completion_times else 0
    avg_ttft = statistics.mean(valid_ttft_times) if valid_ttft_times else 0
    tps = total_tokens / total_time if total_time > 0 else 0
    
    print(f"✅ 完成 Batch={batch_size}. 耗时: {total_time:.2f}s, 总Tokens: {total_tokens}, TPS: {tps:.2f}")
    
    return {
        "batch_size": batch_size,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "tps": tps,
        "avg_ttft": avg_ttft,
        "avg_latency": avg_latency
    }

def main():
    # 1. 环境设置
    default_model_path = "/workspace/Qwen2.5-0.5B-Instruct"
    hf_model = os.environ.get("HF_MODEL", default_model_path)
    os.environ["HF_MODEL"] = hf_model
    print(f"Using model path: {hf_model}")

    # 2. 初始化引擎 (单次初始化，多次运行)
    args = build_args_tt_qwen_batch(hf_model)
    chitu_init(args)
    chitu_start()

    # 3. Warmup (可选，运行一个小任务预热编译缓存)
    print("\n🔥 Pre-warming engine...")
    run_benchmark(batch_size=1, max_new_tokens=10)
    
    # 4. 执行多 Batch 测试
    benchmark_results = []
    # 测试列表，可根据显存和耗时情况调整
    test_batches = [1, 2, 4, 8] 
    
    for bs in test_batches:
        # 稍微暂停，确保之前的任务清理完毕（虽然 TaskPool 逻辑应处理好）
        time.sleep(1)
        res = run_benchmark(batch_size=bs, max_new_tokens=64)
        benchmark_results.append(res)
    
    # 5. 输出对比表格
    print("\n\n" + "="*90)
    print(f"{'Batch':<6} | {'Total Time(s)':<14} | {'Total Tokens':<12} | {'TPS (sys)':<10} | {'TTFT (s)':<10} | {'Avg Latency(s)':<14}")
    print("-" * 90)
    for r in benchmark_results:
        print(f"{r['batch_size']:<6} | {r['total_time']:<14.3f} | {r['total_tokens']:<12} | {r['tps']:<10.2f} | {r['avg_ttft']:<10.3f} | {r['avg_latency']:<14.3f}")
    print("="*90)
    
    # 6. 结束
    chitu_terminate()
    print("\nDone.")

if __name__ == "__main__":
    main()