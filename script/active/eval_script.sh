cores=30
mkdir eval_logs

LOG_SUFFIX="220518-final"
#LOG_SUFFIX="220508-2_src_eval"



#TENT TT_SINGLE_STATS TT_BATCH_STATS Ours
#iabn_k in 3 4 5 7 10 100 1e30
bn_momentum=0.001 #0.1 0.05 0.01 0.001
iabn_k=5          #3 4 5 7 10 100 1e30
#thr=1e-3 #5e-1 1e-2 5e-2 1e-3 5e-3#1e-4 #3 4 5 7 10 100 1e30
for dataset in dogwalk; do #cifar10 cifar100 kitti_sot harth extrasensory reallifehar vlcs pacs hhar wesad ichar icsr
  for method in Src; do                                                               #Src TT_BATCH_STATS PseudoLabel TENT T3A COTTA Ours
    ## main command
    #avg_acc_online
    #avg_f1_online
    if [[ "$method" == *"Src"* ]]; then

      python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/src_user4_train/.*${LOG_SUFFIX}.*

    else
      for dist in 0 1; do
        #        python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_.*s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}.txt &
        if [ "${method}" = "T3A_woBN" ]; then
          python eval_script.py --eval_type avg_acc_online --directory ${dataset}/T3A/ --regex .*${dataset}.*/T3A/.*${LOG_SUFFIX}.*_fuse.*dist${dist}.*s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}.txt &
          #        elif [ "${method}" = "T3A" ]; then
          #          python eval_script.py --eval_type avg_acc_online --directory ${dataset}/T3A/ --regex .*${dataset}.*/T3A/.*${LOG_SUFFIX}.*_nofuse.*dist${dist}.*s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}.txt &
        else

          ###Default, baselines:
#                          python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_dist${dist}\$ 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}.txt &

          ###IABN:
#                                    python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_.*k${iabn_k}.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}.txt &

          ###BN + momentum
#                                    python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_mt${bn_momentum}.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}.txt &
          ###IABN + momentum
                              python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_.*k${iabn_k}_.*mt${bn_momentum}.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}.txt &
          ###Ablation
#                python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_.*mt${bn_momentum}_.*k${iabn_k}_.*${ablation}.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}_${bn_momentum}_${iabn_k}_${ablation}.txt &
        fi
      done

    fi

    ###limit number of processes###
    background=($(jobs -p))
    if ((${#background[@]} >= cores)); then
      wait -n
    fi
    ###############################
  done
done

