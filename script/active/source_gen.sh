cnt=0
for method in Src; do #TENT LAME LAME_vote TENT_vote
  for i in 1 2 3 4 5; do
    for dist in 0; do
        dist=$dist
        update_every_x=16
        memory_size=16
        TGT="user${i}_test"
        METHOD=$method
        EPOCH=50
        LOG_SUFFIX="RESULTS_baseline_source"
        python main.py  --epoch 50 --gpu_idx $((cnt%8)) --model Dogwalk_model_win5 --dataset dogwalk_all_win5 --method Src --validation --tgt ${TGT} --log_suffix ${LOG_SUFFIX} --seed 0 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}.txt &
        cnt=$((cnt + 1))
      done
  done
  wait
done