python -m torch.distributed.run --nproc-per-node=2 examples/recipes/qwen_vl/finetune_qwen_vl.py \
	--pretrained-checkpoint ./checkpoints/qwen3vl_8b \
	--recipe qwen3_vl_8b_finetune_config \
	--dataset-type hf \
	dataset.maker_name=make_cord_v2_dataset \
    model.tensor_model_parallel_size=2 \
	train.global_batch_size=2 \
	train.train_iters=100 \
	logger.wandb_project=megatron_test \
	logger.wandb_save_dir=outputs/wandb/megatron_test \
	checkpoint.save=outputs/megatron_test