# python eval.py \
# --name chair \
# --category chair \
# --checkpoints_dir checkpoints \
# --which_epoch best

python eval.py \
--name chair \
--category chair \
--checkpoints_dir checkpoints \
--batchSize 8 \
--gpu_ids 0 \
--which_epoch best