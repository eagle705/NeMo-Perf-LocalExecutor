# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import basename, splitext

import nemo_run as run
from argument_parser import parse_cli_args
from utils import get_user_configs, hf_tokenizer, set_primary_perf_configs, slurm_executor

from nemo.collections.llm.recipes.llama3_8b import pretrain_recipe
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.lightning.run.plugins import NsysPlugin, PerfEnvPlugin


def override_recipe_configs(
    args: str,
    num_nodes: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
):
    """
    llama3 8b pre-train recipe aimed at achieving best possible performance and faster
    overall runtime.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    recipe = pretrain_recipe(performance_mode=True)
    recipe = set_primary_perf_configs(
        recipe,
        args.tensorboard,
        num_nodes,
        args.gpus_per_node,
        mbs,
        gbs,
        args.max_steps,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
    )

    # data module configs
    recipe.data.num_train_samples = args.max_steps * gbs * mbs  # ensure only 1 epoch for whole run
    recipe.data.tokenizer = hf_tokenizer("meta-llama/Meta-Llama-3-8B")

    # compute dtype configs
    if args.compute_dtype.lower() == "fp8":
        recipe.trainer.plugins = bf16_with_fp8_mixed()
        recipe.trainer.plugins.grad_reduce_in_fp32 = False

    enable_cuda_graph = bool(args.gpu.lower() in [])
    recipe.model.config.enable_cuda_graph = enable_cuda_graph
    recipe.trainer.strategy.use_te_rng_tracker = enable_cuda_graph

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "llama3", "8b", args)
    num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, _ = kwargs

    recipe = override_recipe_configs(args, num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size)

    exp_config = f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_{mbs}mbs_{gbs}gbs"
    exp_name = f"{splitext(basename(__file__))[0]}_{args.compute_dtype}_{exp_config}"

    # executor = slurm_executor(
    #     args.account,
    #     args.partition,
    #     args.log_dir,
    #     num_nodes,
    #     args.gpus_per_node,
    #     args.time_limit,
    #     args.container_image,
    #     custom_mounts=[],
    #     custom_env_vars={},
    #     hf_token=args.hf_token,
    #     nemo_home=args.nemo_home,
    # )
    def local_executor_torchrun(nodes: int = 1, devices: int = 8) -> run.LocalExecutor:
        # Env vars for jobs are configured here
        env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1", # Disable caching NCCL communication buffer memory
        "TRANSFORMERS_OFFLINE": "1", # Enable online downloads from HuggingFace
        "TOKENIZERS_PARALLELISM": "False", # Restrict warning message prints
        "NCCL_NVLS_ENABLE": "0", # Disable NVLink SHARP to save memory
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "1", # Enable cuDNN fused attention
        "NVTE_FLASH_ATTN": "1", # Enable Flash Attention, which is needed to enable cuDNN fused attention
        "NEMO_LOG_MEMORY_USAGE": "1", # Print memory allocation
        "NEMORUN_HOME": args.log_dir,
        "HF_TOKEN": args.hf_token,
        }

        executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

        return executor
    executor = local_executor_torchrun(nodes=1, devices=8)

    plugins = [
        PerfEnvPlugin(enable_vboost=True, nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None),
    ]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=5, end_step=6))

    with run.Experiment(exp_name) as exp:
        exp.add(
            recipe,
            executor=executor,
            name=exp_name,
            plugins=plugins,
        )

        if not args.dryrun:
            # exp.run(sequential=True, detach=True)
            exp.run(sequential=True, tail_logs=True)
        else:
            exp.dryrun()
