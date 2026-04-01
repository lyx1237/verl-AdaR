"""
AdaR Self-Play 训练入口脚本 (run_adar_selfplay.py)

替代 verl.trainer.main_ppo, 使用自定义的 RayAdaRSelfPlayTrainer.

用法:
  python -m recipe.adar_selfplay.run_adar_selfplay \\
      algorithm.adv_estimator=grpo \\
      data.train_files=... \\
      adar_selfplay.enable_selfplay=True \\
      ...
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

from .adar_selfplay_ray_trainer import RayAdaRSelfPlayTrainer


@hydra.main(config_path="config", config_name="adar_selfplay_trainer", version_base=None)
def main(config):
    run_adar_selfplay(config)


def run_adar_selfplay(config) -> None:
    """启动AdaR Self-Play训练"""
    if not ray.is_initialized():
        # 清除RAY_ADDRESS以确保启动本地集群, 避免连接到远程集群
        os.environ.pop("RAY_ADDRESS", None)

        # 确保关键环境变量已设置, 这些变量会被raylet和worker继承
        # (解决flashinfer JIT因CUDA版本不匹配导致编译失败的问题)
        for key in ["CUDA_HOME", "CUDACXX", "VLLM_ATTENTION_BACKEND",
                     "VLLM_USE_FLASHINFER_SAMPLER", "FLASHINFER_ENABLE_AOT"]:
            val = os.environ.get(key, "")
            if val:
                print(f"---INIT--- 环境变量 {key}={val}")

        # 移除/tmp/ray/ray_current_cluster, 避免连接到其他用户的ray集群
        ray_current_cluster = "/tmp/ray/ray_current_cluster"
        if os.path.exists(ray_current_cluster):
            try:
                os.remove(ray_current_cluster)
                print("---INIT--- 已移除stale ray_current_cluster")
            except OSError:
                print("---INIT--- 无法移除ray_current_cluster, 忽略")

        # 传递CUDA和vLLM相关环境变量到Ray worker
        cuda_env = {}
        for key in ["CUDA_HOME", "CUDACXX", "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                     "VLLM_ATTENTION_BACKEND"]:
            if key in os.environ:
                cuda_env[key] = os.environ[key]

        default_runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                **cuda_env,
            }
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        # 强制启动新的本地集群 (不连接已有集群)
        init_kwargs = OmegaConf.to_container(ray_init_kwargs)
        init_kwargs["ignore_reinit_error"] = True
        # 使用随机端口启动新集群, 避免连接到其他用户的ray集群
        import random
        gcs_port = random.randint(20000, 30000)
        init_kwargs["_temp_dir"] = f"/tmp/ray_adar_{os.getpid()}"
        print(f"---INIT--- ray init kwargs: {init_kwargs}")
        print(f"---INIT--- 使用临时目录: {init_kwargs['_temp_dir']}")
        ray.init(**init_kwargs)

    # 保存CUDA环境变量, 传递给TaskRunner
    cuda_env_vars = {}
    for key in ["CUDA_HOME", "CUDACXX", "PATH", "LD_LIBRARY_PATH"]:
        if key in os.environ:
            cuda_env_vars[key] = os.environ[key]

    try:
        runner = TaskRunner.remote()
        ray.get(runner.run.remote(config, cuda_env_vars))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config, cuda_env_vars=None):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        # 恢复CUDA环境变量 (从driver进程传递过来)
        if cuda_env_vars:
            for key, value in cuda_env_vars.items():
                os.environ[key] = value
                print(f"---INIT--- 设置环境变量: {key}={value[:80]}...")

        cuda_home = os.environ.get("CUDA_HOME", "")
        print(f"---INIT--- TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}, CUDA_HOME={cuda_home}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # 下载模型到本地
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # 初始化tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # 定义worker类
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(f"不支持的策略: {config.actor_rollout_ref.actor.strategy}")

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # reward model (可选)
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model (如果使用KL)
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 加载reward函数
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )

        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        # 注册AdaR reward函数 (monkey-patch)
        from .reward_func import register_adar_reward
        register_adar_reward()

        # 创建trainer
        trainer = RayAdaRSelfPlayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        print("---INIT--- 初始化workers...")
        trainer.init_workers()

        print("---INIT--- 开始训练...")
        trainer.fit()

        print("---INIT--- 训练完成!")


if __name__ == "__main__":
    main()
