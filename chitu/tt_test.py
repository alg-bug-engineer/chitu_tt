import os
from types import SimpleNamespace
import sys
sys.path.append("/workspace/chitu_cc")

from chitu.chitu_main import chitu_init, warmup_engine, chitu_start, chitu_run, chitu_terminate
from chitu.backend import Backend
from chitu.task import Task, TaskPool, UserRequest
from chitu.task_type import TaskDecodeType


def build_args_tt_qwen(hf_model_path: str | None = None):
    # Minimal nested args for TT-Qwen path
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
        max_reqs=1,
        max_seq_len=256,
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
    
    # Scheduler 配置（单机单卡场景，使用简单的 fcfs 调度器）
    scheduler = SimpleNamespace(
        type="fcfs",  # First come, first service
    )
    
    # PP 配置（虽然 pp_size=1，但 Scheduler.build 可能会访问）
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


def main():
    # 设置 HF_MODEL 环境变量（TT demo 使用，与 run_demo.py 保持一致）
    # 如果环境变量已设置则使用，否则使用默认路径
    default_model_path = "/workspace/Qwen2.5-0.5B-Instruct"
    hf_model = os.environ.get("HF_MODEL", default_model_path)
    os.environ["HF_MODEL"] = hf_model  # 确保环境变量已设置，供 create_tt_model 使用
    
    print(f"使用模型路径: {hf_model}")

    args = build_args_tt_qwen(hf_model)
    chitu_init(args)
    chitu_start()

    # 构造一次最小请求（单轮，greedy）
    prompt = "介绍下人工智能"
    max_iterations = 128
    
    req = UserRequest(
        message=[{"role": "user", "content": prompt}],
        request_id="tt-demo-1",
        max_new_tokens=max_iterations,
        temperature=0.7,
        top_p=1.0,
        top_k=1,
        logprobs=False,
        top_logprobs=None,
    )
    task = Task(task_id=req.request_id, req=req, stop_with_eos=True)
    TaskPool.add(task)

    # 简单运行循环，直到任务完成
    for iteration in range(max_iterations):
        chitu_run()
        
        # 检查是否有批次结果（原始逻辑）
        if len(Backend.last_batch_results) > 0:
            break
            
        # 额外检查：任务是否已完成（通过 req.completed）
        if hasattr(task.req, 'completed') and task.req.completed.is_set():
            break
            
        # 额外检查：任务状态是否已停止
        if task._decode_status in (TaskDecodeType.Stopped, TaskDecodeType.StopEOS, TaskDecodeType.StopLength):
            break
            
        if iteration >= max_iterations - 1:
            print(f"警告: 达到最大迭代次数 {max_iterations}，任务可能未完成")

    # 打印生成结果
    print("\n" + "="*60)
    print("推理结果")
    print("="*60)
    print(f"提示: {prompt}")
    print(f"请求ID: {req.request_id}")
    
    # 从 Task 获取生成的 token
    if hasattr(task, 'response') and len(task.response) > 0:
        # 获取生成的 token 列表
        response_tokens = task.response.to_tensor().cpu().tolist()
        
        # 使用 tokenizer 解码
        if Backend.tokenizer is not None:
            try:
                generated_text = Backend.tokenizer.decode(response_tokens, skip_special_tokens=True)
                print(f"生成文本: {generated_text}")
                print(f"生成 token 数量: {len(response_tokens)}")
            except Exception as e:
                print(f"解码错误: {e}")
                print(f"生成的 token IDs: {response_tokens}")
        else:
            print(f"生成的 token IDs: {response_tokens}")
            print("警告: Tokenizer 不可用，无法解码文本")
    else:
        print("警告: 未生成任何 token")
    
    # 打印任务状态信息
    print(f"\n任务状态:")
    print(f"  - 完成状态: {task.req.completed.is_set()}")
    print(f"  - 解码状态: {task._decode_status}")
    print(f"  - 生成 token 数: {task.num_new_tokens}")
    if hasattr(task.req, 'finish_reason'):
        print(f"  - 完成原因: {task.req.finish_reason}")
    
    # 终止
    chitu_terminate()
    print("\n完成。")


if __name__ == "__main__":
    main()


