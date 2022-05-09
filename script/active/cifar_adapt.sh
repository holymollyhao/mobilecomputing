#Train

#for i in 1; do
#  SRC="original"
#  TGT="original"
#  CP_PATH="wideresnet28-10"
#  MODEL="wideresnet28-10"
#  TYPE="pretrained"
#  LOG_SUFFIX="220423-${i}_cifar10_src-${SRC}_tgt-${TGT}_lr1e-4_${TYPE}_ep100_uex64_s0"
#  python main.py --gpu_idx $((i%8)) --dataset cifar10 --method Src --src ${SRC} --tgt ${TGT} --log_suffix ${LOG_SUFFIX}  --model ${MODEL} --epoch 100 --update_every_x 64 --seed 0 --load_checkpoint_path ${CP_PATH} 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}.txt &
#done
#wait


## Eval
for method in 'TENT' 'T3A'; do
  SRC="original"
  CP_PATH="wideresnet28-10"
  MODEL="wideresnet28-10"
  TYPE="pretrained"
#  CP_PATH="resnet50"
#  LOG_SUFFIX_SRC="220423-${i}_cifar10_src-${SRC}_tgt-${TGT}_lr1e-4_${TYPE}_ep100_uex64_s0"
#  CP_PATH="log/cifar10/Src/src_${SRC}/tgt_${TGT}/${LOG_SUFFIX_SRC}/cp/cp_last.pth.tar"

  for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    SRC=${SRC}
    if [ "${j}" = 1 ]; then
      TGT="shot_noise-5"
    elif [ "${j}" = 2 ]; then
      TGT="motion_blur-5"
    elif [ "${j}" = 3 ]; then
      TGT="snow-5"
    elif [ "${j}" = 4 ]; then
      TGT="pixelate-5"
    elif [ "${j}" = 5 ]; then
      TGT="gaussian_noise-5"
    elif [ "${j}" = 6 ]; then
      TGT="defocus_blur-5"
    elif [ "${j}" = 7 ]; then
      TGT="brightness-5"
    elif [ "${j}" = 8 ]; then
      TGT="fog-5"
    elif [ "${j}" = 9 ]; then
      TGT="zoom_blur-5"
    elif [ "${j}" = 10 ]; then
      TGT="frost-5"
    elif [ "${j}" = 11 ]; then
      TGT="glass_blur-5"
    elif [ "${j}" = 12 ]; then
      TGT="impulse_noise-5"
    elif [ "${j}" = 13 ]; then
      TGT="contrast-5"
    elif [ "${j}" = 14 ]; then
      TGT="jpeg_compression-5"
    elif [ "${j}" = 15 ]; then
      TGT="elastic_transform-5"
    fi

  LOG_SUFFIX="debug_${method}"
  python main.py --gpu_idx $((j%8)) --online --dataset cifar10 --method ${method} --src ${SRC} --tgt ${TGT} --log_suffix ${LOG_SUFFIX}  --model ${MODEL} --epoch 0 --update_every_x 64 --seed 0 --load_checkpoint_path ${CP_PATH} 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}_${j}.txt &
  done
  wait
done

