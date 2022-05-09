# Train
#for i in 7 8; do
#  SRC="original"
#  TGT="original"
#
#    if [ "${i}" = 7 ]; then
#      CP_PATH="resnet34"
#    elif [ "${i}" = 8 ]; then
#      CP_PATH="resnet18"
#    fi
#
#  TYPE="pretrained"
#  LOG_SUFFIX="220419-${i}_kitti_sot_src-${SRC}_tgt-${TGT}_lr1e-4_${TYPE}_ep100_uex64_s0"
#  python main.py --gpu_idx $((i-5)) --dataset kitti_sot --method Src --src ${SRC} --tgt ${TGT} --log_suffix ${LOG_SUFFIX}  --model KITTI_SOT_model --epoch 100 --update_every_x 64 --seed 0 --load_checkpoint_path ${CP_PATH} 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}.txt &
#done
#wait

## Eval
for i in 7 8; do
  SRC="original"
  TGT="original"
    if [ "${i}" = 7 ]; then
      CP_PATH="resnet34"
      MODEL="resnet34"
    elif [ "${i}" = 8 ]; then
      CP_PATH="resnet18"

      MODEL="resnet18"
    fi



  TYPE="pretrained"
#  CP_PATH="resnet50"
  LOG_SUFFIX_SRC="220419-${i}_kitti_sot_src-${SRC}_tgt-${TGT}_lr1e-4_${TYPE}_ep100_uex64_s0"
  CP_PATH="log/kitti_sot/Src/src_${SRC}/tgt_${TGT}/${LOG_SUFFIX_SRC}/cp/cp_last.pth.tar"

  for j in 1 2 3 4; do
    SRC=${SRC}
    if [ "${j}" = 1 ]; then
      TGT="rain-100-val"
    elif [ "${j}" = 2 ]; then
      TGT="rain-100-tgt"
    elif [ "${j}" = 3 ]; then
      TGT="rain-200-val"
    elif [ "${j}" = 4 ]; then
      TGT="rain-200-tgt"
    fi

  LOG_SUFFIX="220419-${i}_kitti_sot_src-${SRC}_tgt-${TGT}_lr1e-4_${TYPE}_ep0_uex64_s0"
  python main.py --gpu_idx ${j} --dataset kitti_sot --method Src --src ${SRC} --tgt ${TGT} --log_suffix ${LOG_SUFFIX}  --model ${MODEL} --epoch 0 --update_every_x 64 --seed 0 --load_checkpoint_path ${CP_PATH} 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}_${j}.txt &
  done
  wait
done

