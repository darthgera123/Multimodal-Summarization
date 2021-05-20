#!/bin/bash
topk=20
midsave=5
# CUDA_LAUNCH_BLOCKING=1
# python experiment.py --seed 43 --model U \
# --train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 4 --tr bert-large-cased --epochs 5 --tsv \
# --num_features 50 --loadpre ./data/uniter-base.pt --num_pos 6 --contrib --exp U50 --topk $topk

python experiment.py --seed 42 --model O \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 4 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 36  --contrib --exp O36 --topk $topk

# python experiment.py --seed 30 --model X \
# --train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
# --num_features 36 --loadpre ./data/LAST_BX.pth --swa --midsave $midsave --exp X30 --topk $topk

# python experiment.py --seed 45 --model V \
# --train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
# --num_features 100 --loadpre ./data/LAST_BV.pth  --exp V45 --topk $topk

# python experiment.py --seed 147 --model D \
# --train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
# --num_features 36 --loadpre ./data/pytorch_model_11.bin --contrib --exp D36 --topk $topk