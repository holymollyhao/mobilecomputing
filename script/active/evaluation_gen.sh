cores=30
mkdir eval_logs
LOG_SUFFIX="220602-cleaned_ver"

for dataset in dogwalk_all_win5; do
  for method in Src LAME_vote; do
    python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/tgt_user.*/.*${LOG_SUFFIX}.*
    background=($(jobs -p))
    if ((${#background[@]} >= cores)); then
      wait -n
    fi
  done
done

