## Src generation
#DATE="220518-final"
#cnt=0
#
## non-diagonal element
#for epoch in 5 10 50 100; do
#  for i in 1 2 3 4; do
#      for j in 1 2 3 4; do
#        if [ "${i}" != "${j}" ]; then
#          SRC="user${i}_train"
#          TGT="user${j}_train"
#          METHOD="Src"
#          EPOCH=${epoch}
#          LOG_SUFFIX="${DATE}_user${i}_to_${j}_epoch${EPOCH}"
#          python main.py --gpu_idx $((cnt%8)) --epoch ${EPOCH} --model Dogwalk_model --dataset dogwalk --method Src --src ${SRC} --tgt ${TGT} --log_suffix ${LOG_SUFFIX} --seed 0 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}.txt &
#          cnt=$((cnt + 1))
#        else
#          SRC="user${i}_train"
#          TGT="user${i}_test"
#          METHOD="Src"
#          EPOCH=${epoch}
#          LOG_SUFFIX="${DATE}_user${i}_to_${j}_epoch${EPOCH}"
#          python main.py --gpu_idx $((cnt%8)) --epoch ${EPOCH} --model Dogwalk_model --dataset dogwalk --method Src --src ${SRC} --tgt ${TGT} --log_suffix ${LOG_SUFFIX} --seed 0 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}.txt &
#          cnt=$((cnt + 1))
#        fi
#      done
#  done
#  wait
#done


for epoch in 5 10 50 100; do
  for i in 1 2 3 4; do
      for j in 1 2 3 4; do
        if [ "${i}" != "${j}" ]; then
          SRC="user${i}_train"
          TGT="user${j}_train"
          METHOD="Src"
          EPOCH=${epoch}
          LOG_SUFFIX="${DATE}_user${i}_to_${j}_epoch${EPOCH}"
          python main.py --gpu_idx $((cnt%8)) --epoch ${EPOCH} --model Dogwalk_model --dataset dogwalk --method Src --src ${SRC} --tgt ${TGT} --log_suffix ${LOG_SUFFIX} --seed 0 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}.txt &
          cnt=$((cnt + 1))
        else
          SRC="user${i}_train"
          TGT="user${i}_test"
          METHOD="Src"
          EPOCH=${epoch}
          LOG_SUFFIX="${DATE}_user${i}_to_${j}_epoch${EPOCH}"
          python main.py --gpu_idx $((cnt%8)) --epoch ${EPOCH} --model Dogwalk_model --dataset dogwalk --method Src --src ${SRC} --tgt ${TGT} --log_suffix ${LOG_SUFFIX} --seed 0 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}.txt &
          cnt=$((cnt + 1))
        fi
      done
  done
  wait
done

