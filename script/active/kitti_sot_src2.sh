## Train
for i in 4; do
    if [ "${i}" = 4 ]; then

      SRC="2d_detection"
      TGT="2d_detection"
    elif [ "${i}" = 5 ]; then
      SRC="original"
      TGT="original"
    elif [ "${i}" = 6 ]; then
      SRC="original-val"
      TGT="original-val"
    fi

  TYPE="scratch"
  LOG_SUFFIX="220419-${i}_kitti_sot_src-${SRC}_tgt-${TGT}_lr1e-4_${TYPE}_ep100_uex64_s0"
  python main.py --gpu_idx ${i} --dataset kitti_sot --method Src --src ${SRC} --tgt ${TGT} --log_suffix ${LOG_SUFFIX}  --model KITTI_SOT_model --epoch 100 --update_every_x 64 --seed 0 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}.txt &

done
wait

## Eval

for i in 4; do
    if [ "${i}" = 4 ]; then

      SRC="2d_detection"
      TGT="2d_detection"
    elif [ "${i}" = 5 ]; then
      SRC="original"
      TGT="original"
    elif [ "${i}" = 6 ]; then
      SRC="original-val"
      TGT="original-val"
    fi

  TYPE="scratch"
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
  python main.py --gpu_idx ${j} --dataset kitti_sot --method Src --src ${SRC} --tgt ${TGT} --log_suffix ${LOG_SUFFIX}  --model KITTI_SOT_model --epoch 0 --update_every_x 64 --seed 0 --load_checkpoint_path ${CP_PATH} 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}_${j}.txt &
  done
  wait
done

