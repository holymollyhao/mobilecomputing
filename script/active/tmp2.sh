SERVER=$(hostname)

#TODO: num_concurrent python->script
#DATE="220424-1_baselines"
#DATE="220427-2_iabn"
#DATE="220427-3_iabn_k_1m"
#DATE="220427-4_iabn_k_1m_bugfix"
#DATE="220427-5_ours-reprod"
#DATE="220427-6_ours-cbfifo"
#DATE="220427-7_iabn_k_1b_bugfix"
#DATE="220427-8_iabn_k_1b_bugfix_nograd"
#DATE="220428-8_iabn_k_1e20_nograd"
#DATE="220428-1_iabn_k_1e20_nograd_mu-comment"
#DATE="220428-2_iabn_k_1e20_mu-comment"
#DATE="220428-3_iabn_k_1e20_deepcopy"
#DATE="220428-4_iabn_k_1e20_deepcopy_nograd"
#DATE="220428-5_iabn_k_1e20_deepcopy"
#DATE="220428-6_iabn_k_1e20_deepcopy_nograd"
#DATE="220428-7_iabn_k_3_deepcopy"
#DATE="220428-9_iabn_k_2_deepcopy"
#DATE="220428-10_iabn_k_3_eval_src"
#DATE="220428-11_iabn_k_3_eval_ours"

#DATE="220428-12_eval_src"
#DATE="220428-13_ours_no_optim"
#DATE="220428-14_ours_detach_k3"
#DATE="220428-15_ours_detach_k4"
#DATE="220428-16_ours_detach_k1e30"
#DATE="220429-1_cifar100"
#DATE="220428-17_ours_detach_k5"
#DATE="220428-18_ours_detach_k6"
#DATE="220428-19_ours_detach_k10"
#DATE="220429-1_baselines"
#DATE="220429-3_iabn"
#DATE="220430-1_src_iabn"
#DATE="220430-2_ours_no-optim"
#DATE="220430-3_src_iabn_stats"
#DATE="220430-4_src_iabn_bufix"
#DATE="220501-1_src_iabn_thres_bn"
#DATE="220501-2_src_iabn_thres_in"
#DATE="220502-1_bn-stats_iabn"
#DATE="220503-1_skip_thres"
#DATE="220503-2_skip_thres"
#DATE="220503-3_kitti_iabn_fine-tune"
#DATE="220503-4_iabn_trainable"

#DATE="220503-5_kitti_iabn_fine-tune_k3"
#DATE="220503-6_kitti_iabn_fine-tune_k4"
#DATE="220503-7_kitti_iabn_fine-tune_k5"
#DATE="220503-8_kitti_iabn_fine-tune_k10"
#DATE="220504-1_kitti_iabn_fine-tune_k3"
#DATE="220504-2_kitti_iabn_fine-tune_k4"
#DATE="220504-3_kitti_iabn_fine-tune_k5"
#DATE="220504-4_kitti_iabn_fine-tune_k10"

#DATE="220504-5_kitti_iabn_fine-tune_k3-no_optim"
#DATE="220504-6_kitti_iabn_fine-tune_k4-no_optim"
#DATE="220504-7_kitti_iabn_fine-tune_k5-no_optim"
#DATE="220504-8_kitti_iabn_fine-tune_k10-no_optim"

#DATE="220504-9_kitti_iabn_fine-tune_k3-no_detach"
DATE="220504-10_kitti_iabn_fine-tune_k4-no_detach"
#DATE="220504-11_kitti_iabn_fine-tune_k5-no_detach"
#DATE="220504-12_kitti_iabn_fine-tune_k10-no_detach"


#harth reallifehar extrasensory kitti_sot cifar10 cifar100
#Src FT_all CDAN SHOT FeatMatch TT_BATCH_STATS PseudoLabel TENT T3A COTTA
#harth reallifehar extrasensory
if [ "${SERVER}" = "suzy" ]; then
  DATASETS="kitti_sot"
  METHODS="Ours"
elif [ "${SERVER}" = "iu" ]; then
  DATASETS="kitti_sot"
  METHODS="Ours"
elif [ "${SERVER}" = "chris" ]; then
  DATASETS="kitti_sot"
  METHODS="Ours"
elif [ "${SERVER}" = "sooae" ]; then
  DATASETS="kitti_sot"
  METHODS="Ours"
fi

echo SERVER: $SERVER
echo DATASETS: $DATASETS
echo METHODS: $METHODS
#TODO: remove LOG_SUFFIX_SRC. Replace it with a specific pth.
for DATASET in $DATASETS; do # extrasensory reallifehar harth hhar wesad ichar icsr ;opportunity gait hhar  #  #icsr  #ichar  # # # # #
  for METHOD in $METHODS; do #Src FT_all SHOT

    bn_momentum="0.01"
    update_every_x="64"
    memory_size="64"

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
      num_con=1
      MODEL="resnet50"
#      LOG_SUFFIX_SRC="log/kitti_sot/Src/src_2d_detection/tgt_2d_detection/220419-1_kitti_sot_src-2d_detection_tgt-2d_detection_lr1e-4_pretrained_ep100_uex64_s0/cp/cp_last.pth.tar"
#      LOG_SUFFIX_SRC="log/kitti_sot/Src/src_original/tgt_original/220419-2_kitti_sot_src-original_tgt-original_lr1e-4_pretrained_ep100_uex64_s0/cp/cp_last.pth.tar" ########Selected

#      LOG_SUFFIX_SRC="log/kitti_sot/Src/tgt_rain-200/220503-3_kitti_iabn_fine-tune_ep100_uex64_k3_s0/cp/cp_last.pth.tar"
      LOG_SUFFIX_SRC="log/kitti_sot/Src/tgt_rain-200/220503-3_kitti_iabn_fine-tune_ep100_uex64_k4_s0/cp/cp_last.pth.tar"
#      LOG_SUFFIX_SRC="log/kitti_sot/Src/tgt_rain-200/220503-3_kitti_iabn_fine-tune_ep100_uex64_k5_s0/cp/cp_last.pth.tar"
#      LOG_SUFFIX_SRC="log/kitti_sot/Src/tgt_rain-200/220503-3_kitti_iabn_fine-tune_ep100_uex64_k10_s0/cp/cp_last.pth.tar"
      lr="0.0001"

    elif [ "${DATASET}" = "cifar10" ]; then
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



    if [ "${METHOD}" = "Src" ]; then ################ SRC (pretrain)
      SEED="0"

      if [ "${DATASET}" = "kitti_sot" ]; then
        EPOCH=100
      python script.py \
        --num_concurrent 2 \
        --log_as_file \
        --script \
        "--dataset $DATASET \
--model $MODEL \
--method $METHOD \
--epoch $EPOCH \
--seed $SEED \
--log_suffix $LOG_SUFFIX \
--update_every_x ${update_every_x} \
--iabn \
--iabn_k 3 4 5 10 \
--load_checkpoint_suffix resnet50"

      else
        EPOCH=50
      python script.py \
        --num_concurrent 2 \
        --log_as_file \
        --script \
        "--dataset $DATASET \
--model $MODEL \
--method $METHOD \
--epoch $EPOCH \
--seed $SEED \
--update_every_x ${update_every_x} \
--log_suffix $LOG_SUFFIX_SRC"
      fi

    elif
      [ "${METHOD}" = "Src_eval" ]
    then ################ SRC (pretrain)
      LOG_SUFFIX=${DATE}
      SEED="0"
      EPOCH=0
      METHOD="Src"
      iabn="--iabn"
#        iabn="--use_learned_stats"


#        for sigma2_b_thres in 5e-1 5e-2 1e-2 5e-3 1e-3; do
      python script.py \
        --num_concurrent $num_con \
        --log_as_file \
        --script \
        "--dataset $DATASET \
--model $MODEL \
--method $METHOD \
--epoch $EPOCH \
--seed $SEED \
--load_checkpoint_suffix $LOG_SUFFIX_SRC \
--update_every_x ${update_every_x} \
${iabn} \
--iabn_k 3 4 5 10 \
--log_suffix $LOG_SUFFIX"

    elif
      [ "${METHOD}" = "FT_all" ]
    then
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
\
--online \
--epoch 1 \
--update_every_x ${update_every_x} \
--memory_size 200 \
--memory_type FIFO \
\
--log_suffix ${LOG_SUFFIX}"

    elif [ "${METHOD}" = "TT_BATCH_STATS" ]; then
        LOG_SUFFIX=${DATE}
        SEED="0" #"0 1 2"
        nsample=99999
      for iabn_k in 3 4 5 7 10 100 1e30 ; do
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
\
--epoch 1 \
--update_every_x ${update_every_x} \
--memory_size ${memory_size} \
\
--iabn \
--iabn_k ${iabn_k} \
\
--log_suffix ${LOG_SUFFIX}"
done
    elif [ "${METHOD}" = "PseudoLabel" ]; then

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
--update_every_x ${update_every_x} \
--memory_size ${memory_size} \
\
--log_suffix ${LOG_SUFFIX}"
    elif [ "${METHOD}" = "TENT" ]; then

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
--update_every_x ${update_every_x} \
--memory_size ${memory_size} \
\
--log_suffix ${LOG_SUFFIX}"

elif [ "${METHOD}" = "T3A" ]; then

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
--update_every_x ${update_every_x} \
--memory_size ${memory_size} \
\
--log_suffix ${LOG_SUFFIX}"

elif [ "${METHOD}" = "COTTA" ]; then
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
--update_every_x ${update_every_x} \
--memory_size ${memory_size} \
\
\
\
--log_suffix ${LOG_SUFFIX}"

    elif [ "${METHOD}" = "Ours" ]; then
        #default
#        memory_type="CBFIFO"
        memory_type="CBReservoir"
        use_learned_stats="--use_learned_stats"
        temperature="1"
        loss_scaler="1"
        iabn="--iabn"
#        iabn="--use_learned_stats"
#        no_optim="--no_optim"
        no_optim="--online"

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
${iabn} \
--iabn_k 4 \
${no_optim} \
\
--log_suffix ${LOG_SUFFIX}"

    fi
#    python send_email.py --address $ADDR --title ${SERVER}:WWW@${METHOD}_${DATASET}_${DATE}
  done
done
