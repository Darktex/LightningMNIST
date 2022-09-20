python -m lightningmnist.train \
    --trainer.max_epochs=100 \
    --trainer.accelerator="gpu" \
    --trainer.num_nodes=1 \
    --trainer.devices=2 \
    --trainer.strategy="ddp_find_unused_parameters_false"
