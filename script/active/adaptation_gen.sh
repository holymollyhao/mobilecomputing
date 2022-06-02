DATE="220602-cleaned_ver"
cnt=0
for method in LAME_vote; do #TENT LAME LAME_vote TENT_vote
  for i in 1 2 3 4 5; do
    for dist in 0; do
        dist=$dist
        update_every_x=64
        memory_size=64
        TGT="user${i}_test"
        METHOD=$method
        EPOCH=${epoch}
        LOG_SUFFIX="${DATE}_epoch${EPOCH}_uex${update_every_x}_memsize${memory_size}_dist${dist}"
        CHECKPOINT=./log/dogwalk_all_win5/Src/tgt_${TGT}/baseline_source/cp/cp_last.pth.tar
        python main.py --tgt_train_dist $dist --load_checkpoint_path $CHECKPOINT --lr 0.0001 --epoch 1 --online --update_every_x $update_every_x --memory_size $memory_size --gpu_idx $((cnt%8)) --model Dogwalk_model_win5 --dataset dogwalk_all_win5 --method $METHOD --validation --tgt ${TGT} --log_suffix ${LOG_SUFFIX} --seed 0 2>&1 | tee raw_logs/${LOG_SUFFIX}_job${i}.txt &
        cnt=$((cnt + 1))
      done
  done
  wait
done