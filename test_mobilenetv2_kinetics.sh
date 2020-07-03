python tools/run_net.py --cfg ./configs/Kinetics/mobilenetv2.yaml \
    OUTPUT_DIR ./logs/kinetics-mobilenetv2-8x8

python tools/run_net.py --cfg ./configs/Kinetics/mobilenetv2.yaml \
    MOBILENET.WIDTH_MULT 0.8 \
    OUTPUT_DIR ./logs/kinetics-mobilenetv2-0.8-8x8

