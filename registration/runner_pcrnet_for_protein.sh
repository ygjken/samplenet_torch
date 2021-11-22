# Train PCR-Net
python main.py \
    -o log/for_protein/PCRNet1024 \
    --datafolder data/protein \
    --sampler none \
    --train-pcrnet \
    --epochs 100 \
    --protein

# Test PCR-Net
 python main.py \
    -o log/for_protein/PCRNet1024 \
    --datafolder data/protein \
    --sampler none \
    --test \
    --epochs 100 \
    --protein \
    --transfer-from log/baseline/PCRNet1024_model_best.pth