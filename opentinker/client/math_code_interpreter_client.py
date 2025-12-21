#!/usr/bin/env python3
"""
Example: Math Training with Code Interpreter (Multi-Turn Agent Loop)

This client trains an LLM to solve math problems using Python code execution.
Uses agent_loop algorithm with GymEnvironmentInteraction - the code execution
happens in the game server (CodeInterpreterMathGame).

Architecture:
    Client
      ↓ submit job
    Training Server (GenericAgentLoop)
      ↓ HTTP via GymEnvironmentInteraction  
    Game Server (CodeInterpreterMathGame)
      ↓ internal HTTP call
    Sandbox Server (code execution)

Usage:
    1. Start the scheduler:
       python opentinker/scheduler/launch_scheduler.py
       
    2. Start the game server (includes sandbox):
       python opentinker/environment/math/code_interpreter_math_server.py --port 8088
       
    3. Run this client:
       python math_code_interpreter_client.py \\
           data_path=/path/to/train.parquet \\
           val_data_path=/path/to/test.parquet \\
           tokenizer_path=/path/to/model
"""

import hydra
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader

from http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.base_game_environment import GameEnvironment
from opentinker.environment.base_data_generator import DynamicGameDataset, collate_fn
from opentinker.environment.math.code_interpreter_math import CodeInterpreterMathGame
from opentinker.environment.static_data_generator import StaticDatasetGenerator
from opentinker.environment.game_stats_client import GameStatsClient
from utils import resolve_paths_in_config
from scheduler_client_lifecycle import get_lifecycle_manager
from verl.trainer.main_ppo import create_rl_sampler
from opentinker.environment.math.math_code_interpreter_env import MathCodeInterpreterEnvironment

@hydra.main(config_path="client_config", config_name="math_code_interpreter_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()
    
    print("=" * 60)
    print("Math Training with Code Interpreter (Agent Loop)")
    print("=" * 60)
    
    # 1. Submit job to scheduler
    print("\n[1/4] Submitting job to scheduler...")
    scheduler_client = SchedulerClient(
        scheduler_url=args.get("scheduler_url", "http://localhost:8780"),
        api_key=args.get("scheduler_api_key")
    )
    
    job_result = scheduler_client.submit_job(
        config=OmegaConf.to_container(args, resolve=True),
        enable_agent_loop=True,
        wandb_key=args.get("wandb_key"),
        num_gpus=args.get("num_gpus"),
    )
    
    job_id = job_result["job_id"]
    server_url = job_result["server_url"]
    lifecycle.register_job(scheduler_client, job_id)
    
    print(f"✓ Job {job_id} allocated at {server_url}")
    
    # 2. Setup environment
    print("\n[2/4] Setting up environment...")
    env_endpoint = args.interaction.config.env_endpoint
    env = MathCodeInterpreterEnvironment(
        game_class=CodeInterpreterMathGame,
        config=args,
        data_paths=[args.data_path],
        val_data_paths=[args.val_data_path] if args.val_data_path else None,
        job_id=job_id,
    )
    print(f"✓ Environment created")
    print(f"  - Interaction config: {env.get_interaction_config_path()}")
    print(f"  - Game server endpoint: {env_endpoint}")
    
    # 3. Setup game stats client
    print("\n[3/4] Connecting to game server...")
    game_stats = GameStatsClient(env_endpoint, job_id=env.job_id)
    if game_stats.health_check():
        game_stats.reset_all()
        print(f"✓ Connected to game server at {env_endpoint}")
    else:
        game_stats = None
        print(f"⚠ Game server not responding at {env_endpoint}")
        print(f"  Make sure to start: python opentinker/environment/math/code_interpreter_math_server.py --port {args.interaction.config.env_port}")
    
    # 4. Connect to training server and train
    print("\n[4/4] Starting training...")
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )
    client.set_config(args, env)
    
    print(f"\nTraining configuration:")
    print(f"  - Algorithm: {args.algorithm}")
    print(f"  - Steps: {args.get('num_steps')}")
    print(f"  - Epochs: {args.get('num_epochs')}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max turns: {args.multi_turn.max_assistant_turns}")
    
    try:
        final_metrics = client.fit(
            env=env,
            num_epochs=args.get("num_epochs"),
            num_steps=args.get("num_steps"),
            save_freq=args.save_freq,
            test_freq=args.test_freq,
            verbose=True,
            validate_before_training=True,
            game_stats_client=game_stats,
        )
        print(f"\n✓ Training completed!")
        print(f"Final metrics: {final_metrics}")
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()
