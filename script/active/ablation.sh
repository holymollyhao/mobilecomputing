SERVER=$(hostname)

#DATE="220424_ablation"
DATE="220429-2_ablation"

#harth reallifehar extrasensory kitti_sot cifar10

if [ "${SERVER}" = "suzy" ]; then
  DATASETS="cifar10"
  METHODS="Ours"
elif [ "${SERVER}" = "iu" ]; then
  DATASETS="cifar100"
  METHODS="Ours"
elif [ "${SERVER}" = "chris" ]; then
  DATASETS="cifar100"
  METHODS="Ours"
elif [ "${SERVER}" = "sooae" ]; then
  DATASETS="harth"
  METHODS="Ours"
fi

echo SERVER: $SERVER
echo DATASETS: $DATASETS
echo METHODS: $METHODS

for DATASET in $DATASETS; do # extrasensory reallifehar harth hhar wesad ichar icsr ;opportunity gait hhar  #  #icsr  #ichar  # # # # #
  for METHOD in $METHODS; do #Src FT_all SHOT

    if [ "${DATASET}" = "harth" ]; then
      MODEL="HARTH_model"
      LOG_SUFFIX_SRC="220327-1_harth_minmax_scaling_all_split_win50"       # src-back, tgt-thigh ########### SELECTED
#      LOG_SUFFIX_SRC="220327-2_harth_minmax_scaling_all_split_win50_val" # src-back, tgt-thigh ########### SELECTED

      lr="0.01"
      num_con=3
    elif [ "${DATASET}" = "extrasensory" ]; then
      num_con=3
      MODEL="ExtraSensory_model"

      LOG_SUFFIX_SRC="220327-1_extrasensory_selectedfeat_woutloc_std_scaling_all_win5"       ############ SELECTED
#      LOG_SUFFIX_SRC="220327-2_extrasensory_selectedfeat_woutloc_std_scaling_all_win5_val" ############ SELECTED

      lr="0.01"
    elif [ "${DATASET}" = "reallifehar" ]; then
      num_con=3
      MODEL="RealLifeHAR_model"
      LOG_SUFFIX_SRC="220327-1_reallifehar_acc_minmax_scaling_all_win400_overlap0"       ############ SELECTED
#      LOG_SUFFIX_SRC="220327-2_reallifehar_acc_minmax_scaling_all_win400_overlap0_val" ############ SELECTED

      lr="0.0001"
    elif [ "${DATASET}" = "kitti_sot" ]; then
      num_con=3
      MODEL="resnet50"
#      LOG_SUFFIX_SRC="log/kitti_sot/Src/src_2d_detection/tgt_2d_detection/220419-1_kitti_sot_src-2d_detection_tgt-2d_detection_lr1e-4_pretrained_ep100_uex64_s0/cp/cp_last.pth.tar"
      LOG_SUFFIX_SRC="log/kitti_sot/Src/src_original/tgt_original/220419-2_kitti_sot_src-original_tgt-original_lr1e-4_pretrained_ep100_uex64_s0/cp/cp_last.pth.tar" ########Selected

      lr="0.0001"

    elif [ "${DATASET}" = "cifar10" ]; then
      num_con=3
      MODEL="wideresnet28-10"
#      LOG_SUFFIX_SRC="log/cifar10/Src/src_original/tgt_original/220421-1_cifar10_src-original_tgt-original_lr1e-4_pretrained_ep100_uex64_s0/cp/cp_last.pth.tar"
      LOG_SUFFIX_SRC="wideresnet28-10" ### SELECTED

      lr="0.0001"
    elif [ "${DATASET}" = "cifar100" ]; then
      num_con=3
      MODEL="resnext29"
#      LOG_SUFFIX_SRC="log/cifar10/Src/src_original/tgt_original/220421-1_cifar10_src-original_tgt-original_lr1e-4_pretrained_ep100_uex64_s0/cp/cp_last.pth.tar"
      LOG_SUFFIX_SRC="resnext29" ### SELECTED

      lr="0.0001"
    fi

    for lr in 0.0001; do #.01 0.001 0.0001
      for bn_momentum in 0.01; do # 0.1 0.01 0.001
        for ablation in case1 case2 case3 case4 case5 case6 case7 case8; do #
          #default
#          lr="0.0001"
          temperature="1"
          loss_scaler="1"
#          bn_momentum="0.01"
          update_every_x="64"
          memory_size="64"
          use_learned_stats="--use_learned_stats"
          no_optim="--online" # placeholder
          update_all="--online" # placeholder
          iabn="--online" #placeholder

          if [ "${ablation}" = "case1" ]; then ####################### optimization
            memory_type="FIFO"
            no_optim="--no_optim"
          elif [ "${ablation}" = "case2" ]; then
            memory_type="CBReservoir"
            no_optim="--no_optim"
          elif [ "${ablation}" = "case3" ]; then
            memory_type="CBReservoir"
            loss_scaler="0"
          elif [ "${ablation}" = "case4" ]; then # current method
            memory_type="CBReservoir"

          elif [ "${ablation}" = "case5" ]; then ######################## memory
            memory_type="FIFO"
          elif [ "${ablation}" = "case6" ]; then
            memory_type="Reservoir"
          elif [ "${ablation}" = "case7" ]; then
            memory_type="CBFIFO"

          elif [ "${ablation}" = "case8" ]; then ######################## layer
            update_all="--update_all"

          elif [ "${ablation}" = "case9" ]; then ######################## iabn
            iabn="--iabn"
          fi

          if [ "${DATASET}" = "harth" ]; then
            :
          elif [ "${DATASET}" = "extrasensory" ]; then
            :
          elif [ "${DATASET}" = "reallifehar" ]; then
            :
          elif [ "${DATASET}" = "kitti_sot" ]; then
            :
          fi

          LOG_SUFFIX=${DATE}
          SEED="0" #"0 1 2"
          nsample=99999
          python script.py \
            --num_concurrent $num_con \
            --log_as_file \
            --script_seed 0 \
            --script \
            "--dataset $DATASET \
--model $MODEL \
--method $METHOD \
--remove_cp \
--seed $SEED \
--load_checkpoint_suffix $LOG_SUFFIX_SRC \
--tgt_train_dist 0 1 \
--nsample $nsample \
--online \
\
--lr ${lr} \
--temperature ${temperature} \
--loss_scaler ${loss_scaler} \
--bn_momentum ${bn_momentum} \
--update_every_x ${update_every_x} \
--memory_size ${memory_size} \
--memory_type ${memory_type} \
${use_learned_stats} \
${no_optim} \
${update_all} \
${iabn} \
\
\
--log_suffix ${LOG_SUFFIX}_${ablation}_lr${lr}_bn${bn_momentum}"
        done
      done
    done
  done
done
