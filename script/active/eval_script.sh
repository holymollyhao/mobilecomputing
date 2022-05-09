cores=30
mkdir eval_logs
###### Online, eval_online

#LOG_SUFFIX="220325-2"
#LOG_SUFFIX="220328_tuned_full"
#LOG_SUFFIX="220328-2_adapt_then_eval"
#LOG_SUFFIX="220329_temp_bugfix"
#LOG_SUFFIX="220402_no_div"
#LOG_SUFFIX="220402-2_no_div_min_loss_1st"
#LOG_SUFFIX="220402-3_no_div_max_loss_1st"
#LOG_SUFFIX="220419-5"
#LOG_SUFFIX="220420-1"
#LOG_SUFFIX="220420-3_cotta_plus_cbfifo_bnmoment_hparams-from-ours"
#LOG_SUFFIX="220422-1_t3a"
#LOG_SUFFIX="220422_ablation_case7"
#LOG_SUFFIX="220422-2_baseline"
#LOG_SUFFIX="220424_ablation_${ablation}_lr${lr}_bn${bn_momentum}"
#LOG_SUFFIX="220423-1_scaler1"
#LOG_SUFFIX="220423-2_scaler1_last-bn"
#LOG_SUFFIX="220424_ablation"
#LOG_SUFFIX="220424-1_baselines"
#LOG_SUFFIX="220427-1_iabn"
#LOG_SUFFIX="220427-2_iabn"
#LOG_SUFFIX="220427-3_iabn_k_1m"
#LOG_SUFFIX="220427-4_iabn_k_1m_bugfix"
#LOG_SUFFIX="220427-5_ours-reprod"
#LOG_SUFFIX="220427-6_ours-cbfifo"
#LOG_SUFFIX="220427-7_iabn_k_1b_bugfix"
#LOG_SUFFIX="220428-7_iabn_k_3_deepcopy"
#LOG_SUFFIX="220428-9_iabn_k_2_deepcopy"
#LOG_SUFFIX="220428-10_iabn_k_3_eval_src"
#LOG_SUFFIX="220428-12_eval_src"
#LOG_SUFFIX="220428-11_iabn_k_3_eval_ours"
#LOG_SUFFIX="220428-13_ours_no_optim"
#LOG_SUFFIX="220429-1_cifar100"
#LOG_SUFFIX="220428-14_ours_detach_k3"
#LOG_SUFFIX="220428-15_ours_detach_k4"
#LOG_SUFFIX="220428-16_ours_detach_k1e30"
#LOG_SUFFIX="220428-17_ours_detach_k5"
#LOG_SUFFIX="220428-18_ours_detach_k6"
#LOG_SUFFIX="220428-19_ours_detach_k10"
#LOG_SUFFIX="220428-10_iabn_k_3_eval_src"
#LOG_SUFFIX="220429-1_baselines"
#LOG_SUFFIX="220429-3_iabn"
#LOG_SUFFIX="220429-2_ablation"
#LOG_SUFFIX="220430-1_src_iabn"
#LOG_SUFFIX="220430-2_ours_no-optim"
#LOG_SUFFIX="220430-4_src_iabn_bufix"
#LOG_SUFFIX="220501-1_src_iabn_thres_bn"
#LOG_SUFFIX="220501-2_src_iabn_thres_in"
#LOG_SUFFIX="220502-1_bn-stats_iabn"
#LOG_SUFFIX="220503-1_skip_thres"
#LOG_SUFFIX="220503-2_skip_thres"
#LOG_SUFFIX="220503-4_iabn_trainable"
#LOG_SUFFIX="220503-5_kitti_iabn_fine-tune_k3"
#LOG_SUFFIX="220503-6_kitti_iabn_fine-tune_k4"
#LOG_SUFFIX="220503-7_kitti_iabn_fine-tune_k5"
#LOG_SUFFIX="220503-8_kitti_iabn_fine-tune_k10"
#LOG_SUFFIX="220503-9_iabn_trainable_scale10"
#LOG_SUFFIX="220503-10_iabn_trainable_scale100"
#LOG_SUFFIX="220503-11_iabn_trainable_scale1000"
#LOG_SUFFIX="220504-1_kitti_iabn_fine-tune_k3"
#LOG_SUFFIX="220504-2_kitti_iabn_fine-tune_k4"
#LOG_SUFFIX="220504-3_kitti_iabn_fine-tune_k5"
#LOG_SUFFIX="220504-4_kitti_iabn_fine-tune_k10"
#LOG_SUFFIX="220504-5_kitti_iabn_fine-tune_k3-no_optim"
#LOG_SUFFIX="220504-6_kitti_iabn_fine-tune_k4-no_optim"
#LOG_SUFFIX="220504-7_kitti_iabn_fine-tune_k5-no_optim"
#LOG_SUFFIX="220504-8_kitti_iabn_fine-tune_k10-no_optim"
#LOG_SUFFIX="220504_src_ep50"
#LOG_SUFFIX="220505-1_src_eval"
#LOG_SUFFIX="220505-2_ours_eval_optim"
#LOG_SUFFIX="220505-3_ours_eval_no-optim"
#LOG_SUFFIX="220505-4_ours_eval_optim"
#LOG_SUFFIX="220505-5_ours_eval_no-optim"
#LOG_SUFFIX="220505-6_ours_eval_optim"
#LOG_SUFFIX="220505-7_ours_eval_no-optim"
#LOG_SUFFIX="220505-8_bn"
#LOG_SUFFIX="220505_src_ep50"
#LOG_SUFFIX="220505-9_eval"
#LOG_SUFFIX="220505-2_src_ep50_r18_pretrained"
#LOG_SUFFIX="220505-3_src_ep100_r18_scratch"
#LOG_SUFFIX="220505-10_src_ep50_r18_pretrained-b128"
#LOG_SUFFIX="220505-10_src_ep50_r18_pretrained-b256"
#LOG_SUFFIX="220505_src_ep100_coslr"
#LOG_SUFFIX="220506_ours-no_optim"
#LOG_SUFFIX="220506_baselines"

#LOG_SUFFIX="220506-1_src_2d_object_eval_baseline_ours-no_optim"
#LOG_SUFFIX="220507-3_eval_t3a_cotta"
#LOG_SUFFIX="220507-2_eval"
#LOG_SUFFIX="220507-4_cotta_debug"
#LOG_SUFFIX="220507-5_cotta_debug_wideresnet28"
#LOG_SUFFIX="220507_ablation"
#LOG_SUFFIX="220507-7_cotta_debug_every200"
#LOG_SUFFIX="220508_vcls_debug"
#LOG_SUFFIX="220508_pacs_debug"

#LOG_SUFFIX="220508_ablation_k4"
LOG_SUFFIX="220508_baselines"
#LOG_SUFFIX="220508-2_src_eval"



#TENT TT_SINGLE_STATS TT_BATCH_STATS Ours
#iabn_k in 3 4 5 7 10 100 1e30
bn_momentum=0.001 #0.1 0.05 0.01 0.001
iabn_k=5          #3 4 5 7 10 100 1e30
#thr=1e-3 #5e-1 1e-2 5e-2 1e-3 5e-3#1e-4 #3 4 5 7 10 100 1e30
sth=3136 # 64 256 3136 784
#for lr in 0.01 0.001 0.0001; do
#for bn_momentum in 0.01; do
#  for iabn_k in 4; do
#    for ablation in case2 case3 case4 case5 case6 case9 case10 case11; do                    # #harth extrasensory reallifehar cifar10 cifar100
      for dataset in pacs; do #cifar10 cifar100 kitti_sot harth extrasensory reallifehar vlcs pacs hhar wesad ichar icsr
        for method in Ours; do                                                               #Src TT_BATCH_STATS PseudoLabel TENT T3A COTTA Ours
          ## main command
          #avg_acc_online
          #avg_f1_online
          if [[ "$method" == *"Src"* ]]; then
            #      if [ "${dataset}" = "vlcs" ] || [ "${dataset}" = "officehome" ] || [ "${dataset}" = "pacs" ]; then
            #        if [ "${method}" = "Src_woBN" ]; then
            #          python eval_script.py --eval_type avg_acc_online --directory ${dataset}/Src/ --regex .*${dataset}.*/Src/.*${DATE}.*fused.* 2>&1 | tee eval_logs/log_${dataset}_${method}.txt &
            #        else
            #          python eval_script.py --eval_type avg_acc_online --directory ${dataset}/Src/ --regex .*${dataset}.*/Src/.*${DATE}.* 2>&1 | tee eval_logs/log_${dataset}_${method}.txt &
            #        fi
            #      else
            ### Default:
                    python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}\$ 2>&1 | tee eval_logs/log_${dataset}_${method}.txt &
            ### IABN:
#                            python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*k${iabn_k}.* 2>&1 | tee eval_logs/log_${dataset}_${method}.txt &
            ###Ablation
#            python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_.*mt${bn_momentum}_.*k${iabn_k}_.*${ablation}.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}_${bn_momentum}_${iabn_k}_${ablation}.txt &
            #      fi
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
#    done
#  done
#done

#
##SHOT CDAN FeatMatch
##TT_SINGLE_STATS TENT VOTE
##Src TT_SINGLE_STATS TT_BATCH_STATS TENT VOTE
##Src CDAN SHOT FeatMatch TT_SINGLE_STATS TT_BATCH_STATS TT_BATCH_PARAMS TT_WHOLE TENT VOTE
#for dataset in harth reallifehar extrasensory; do #extrasensory reallifehar harth hhar wesad ichar icsr
#  for method in Src TENT Ours; do #Src FT_all TT_SINGLE_STATS CDAN SHOT FeatMatch TT_BATCH_STATS TT_BATCH_PARAMS TT_WHOLE VOTE
#        ## main command
#
#      if [ "${method}" = "Src" ]; then
#          python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}.txt &
#
#      elif [ "${method}" = "CDAN" ]  || [ "${method}" = "SHOT" ] || [ "${method}" = "FeatMatch" ]; then
#        for dist in 0 1; do
#          for ep in 1; do
#            for uex in 200; do
#              python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_.*ep${ep}_.*uex${uex}.*s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}_${ep}_${uex}.txt &
#            done
#          done
#        done
#      elif [ "${method}" = "TT_BATCH_STATS" ]  || [ "${method}" = "VOTE" ]; then
#        for dist in 0 1; do
#          for ep in 1; do
#            for uex in 16 32 64; do
#              python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_.*ep${ep}_.*uex${uex}.*s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}_${ep}_${uex}.txt &
#            done
#          done
#        done
#      elif [ "${method}" = "TT_SINGLE_STATS" ]; then
#
#        for dist in 0 1; do
#          for momentum in 0.1 0.01 0.001 0.0001 0.00001 0.000001; do
#            python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_.*bnm${momentum}_.*s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}_${momentum}.txt &
#          done
#        done
#
#      elif [ "${method}" = "TENT_STATS" ] ; then
#
#        for dist in 0 1; do
#          for ep in 1; do
#            for uex in 16 32 64; do
#              for momentum in 0.1; do
#                python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_.*ep${ep}_.*uex${uex}_.*bnm${momentum}_.*s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}_${ep}_${uex}_${momentum}.txt &
#              done
#            done
#          done
#        done
#      elif [ "${method}" = "Ours" ] ; then
#
#        for dist in 0 1; do
#          for ep in 1; do
#            for uex in 16 32 64; do
#              for mt in CBFIFO; do
#                for momentum in 0.1 0.01; do
#                  python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_.*ep${ep}_.*uex${uex}_.*mt${mt}_.*bnm${momentum}_.*s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}_${ep}_${uex}_${mt}_${momentum}.txt &
#                done
#              done
#            done
#          done
#        done
#      else
#        for dist in 0 1; do
#          for ep in 1; do
#            for uex in 64; do
#              python eval_script.py --eval_type avg_acc_online --directory ${dataset}/${method}/ --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}_.*dist${dist}_.*ep${ep}_.*uex${uex}.*s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}_${dist}_${ep}_${uex}.txt &
#            done
#          done
#        done
#
#      fi
#
#        ###limit number of processes###
#        background=($(jobs -p))
#        if ((${#background[@]} >= cores)); then
#          wait -n
#        fi
#        ###############################
#  done
#done

###### Online
#LOG_SUFFIX=220119
#for dataset in reallifehar harth extrasensory; do #extrasensory reallifehar harth hhar wesad ichar icsr
#  for method in SHOT FT_all CDAN; do #SHOT FT_all CDAN
#    if [ "${method}" = "FT_all" ]; then
#      uex_range="50 100 99999"
#    elif [ "${method}" = "SHOT" ]; then
#      uex_range="100 200 99999"
#    elif [ "${method}" = "CDAN" ]; then
#      uex_range="100 200 99999"
#    fi
#    for dist in 0 1; do
#      for mt in FIFO CBRS; do
#        for mem in 200 500; do
#          for uex in $uex_range; do
#            for ep in 1 5; do
#
#              ## main command
#              python eval_script.py --eval_type avg_acc --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}.*dist${dist}_.*ep${ep}_.*uex${uex}_.*mem${mem}_.*mt${mt}_s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}_dist${dist}_ep${ep}_uex${uex}_mem${mem}_mt${mt}.txt &
#
#              ###limit number of processes###
#              background=($(jobs -p))
#              if ((${#background[@]} == cores)); then
#                wait -n
#              fi
#              ###############################
#            done
#          done
#        done
#      done
#    done
#  done
#done

###### Offline
#LOG_SUFFIX="220125_no-online_no"
#for dataset in harth extrasensory reallifehar; do #extrasensory reallifehar harth hhar wesad ichar icsr
#  for method in CDAN; do #SHOT FT_all CDAN
#    for dist in 0; do
#      for ep in 50; do
#
#        ## main command
#        python eval_script.py --eval_type avg_acc --regex .*${dataset}.*/${method}/.*${LOG_SUFFIX}.*dist${dist}_.*ep${ep}_.*s0.* 2>&1 | tee eval_logs/log_${dataset}_${method}_dist${dist}_ep${ep}.txt &
#
#        ###limit number of processes###
#        background=($(jobs -p))
#        if ((${#background[@]} == cores)); then
#          wait -n
#        fi
#        ###############################
#      done
#    done
#  done
#done
