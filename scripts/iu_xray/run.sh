seed=${RANDOM}
noamopt_warmup=1000

python train.py \
    --image_dir=data/iu_xray/images/ \
    --ann_path=data/iu_xray/annotation.json \
    --dataset_name=iu_xray \
    --max_seq_length=60 \
    --threshold=3 \
    --batch_size=16 \
    --epochs=100 \
    --save_dir=results/iu_xray/base_cmn \
    --step_size=1 \
    --gamma=0.8 \
    --seed=${seed} \
    --topk=32 \
    --beam_size=3 \
    --log_period=100 \
    --noamopt_warmup=${noamopt_warmup}
