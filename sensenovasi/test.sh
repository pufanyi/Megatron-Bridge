python -m torch.distributed.run --nproc-per-node=2 examples/recipes/qwen_vl/finetune_qwen_vl.py \
	--pretrained-checkpoint ./checkpoints/qwen3vl_8b \
	--recipe qwen3_vl_8b_finetune_config \
	--dataset-type preloaded \
	dataset.train_data_path=/mnt/aigc/users/pufanyi/workspace/rot-data/data/data/data/MindCube_train_instructions_qa.jsonl \
    dataset.image_folder=/mnt/aigc/users/pufanyi/workspace/rot-data/data/data/data \
    model.tensor_model_parallel_size=2 \
	train.global_batch_size=16 \
	train.train_iters=10000 \
	logger.wandb_project=megatron_test \
	logger.wandb_save_dir=outputs/wandb/megatron_test \
	checkpoint.save=outputs/megatron_test