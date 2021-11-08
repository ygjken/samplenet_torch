# Train PCR-Net
python main.py \
    -o log/for_protein/PCRNet1024 \
    --datafolder data/protein \
    --sampler none \
    --train-pcrnet \
    --epochs 100 \
    --protein