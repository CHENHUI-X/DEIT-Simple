python main.py \
        --model-ema \
        --clip-grad 10.0  \
        --train-mode  \
        --distillation-type 'soft'  \
        --output_dir ./output/  \
        --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth