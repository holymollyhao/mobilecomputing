SERVER=$(hostname)

#DATE="220327_tuned"
#DATE="220328-2_adapt_then_eval"
#DATE="220329_temp_bugfix"
#DATE="220402_no_div"
#DATE="220402-2_no_div_min_loss_1st"
#DATE="220402-3_no_div_max_loss_1st"
#DATE="220419-2"
#DATE="220419-3"
#DATE="220419-4"
#DATE="220419-5"
#DATE="220420-2"
#DATE="220420-3_cotta_plus_cbfifo_bnmoment_hparams-from-ours"
#DATE="220421-1_cotta_plus_cbfifo_bnmoment_hparams-tuned"
#DATE="220422_ablation"
#DATE="220422-2_baseline"
#DATE="220423-1_scaler1"
#DATE="220423-2_scaler1_last-bn"
#DATE="220424-1_baselines"
DATE="220427-1_iabn"

#harth reallifehar extrasensory kitti_sot cifar10
#Src FT_all CDAN SHOT FeatMatch TT_BATCH_STATS TENT T3A COTTA

if [ "${SERVER}" = "suzy" ]; then
  DATASETS="cifar10 kitti_sot harth reallifehar extrasensory"
  METHODS="Ours"
elif [ "${SERVER}" = "iu" ]; then
  DATASETS="cifar10"
  METHODS="T3A"
elif [ "${SERVER}" = "chris" ]; then
  DATASETS="kitti_sot harth reallifehar extrasensory"
  METHODS="TT_BATCH_STATS TENT T3A COTTA"
elif [ "${SERVER}" = "sooae" ]; then
  DATASETS="harth"
  METHODS="Src T3A"
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
    elif [ "${DATASET}" = "kitti_mot" ]; then
      MODEL="KITTI_MOT_model"

#      LOG_SUFFIX_SRC="220410_kitti_mot_src-original_tgt-rain"       ############ SELECTED # all class

#      LOG_SUFFIX_SRC="220411-1_kitti_mot_src-original_tgt-rain_car_ped_cycl"       ############ SELECTED # 'Car', 'Pedestrian', 'Cyclist'
#      LOG_SUFFIX_SRC="220411-2_kitti_mot_src-original_tgt-rain_car_ped_cycl_imagenet-pretraind"       ############ SELECTED # 'Car', 'Pedestrian', 'Cyclist'

#      LOG_SUFFIX_SRC="220412-1_kitti_mot_src-rain_tgt-original_car_ped_cycl_imagenet-pretraind"       ############ SELECTED # 'Car', 'Pedestrian', 'Cyclist'
#      --load_checkpoint_path = /home/tsgong/git/WWW/log/imagenet/darknet53.conv.74
#      LOG_SUFFIX_SRC="220413-1_kitti_mot_src-original_tgt-rain-100_lr1e-3"
#      LOG_SUFFIX_SRC="220413-2_kitti_mot_src-original_tgt-rain-100_lr1e-4"
#      LOG_SUFFIX_SRC="220413-3_kitti_mot_src-original_tgt-rain-100_lr1e-4_scratch"
#      LOG_SUFFIX_SRC="220413-4_kitti_mot_src-original_tgt-rain-100_lr1e-4_darknet"
#      LOG_SUFFIX_SRC="220414-1_kitti_mot_src-2d_obj_tgt-rain-100_lr1e-4_scratch"
#      LOG_SUFFIX_SRC="220414-2_kitti_mot_src-2d_obj_tgt-rain-100_lr1e-4_darknet"
#      LOG_SUFFIX_SRC="220417-1_kitti_mot_src-2d_obj_tgt-rain-100_lr1e-4_scratch_noaug"
#      LOG_SUFFIX_SRC="220417-2_kitti_mot_src-original-val_tgt-rain-100_lr1e-4_scratch_noaug"
      LOG_SUFFIX_SRC="220417-3_kitti_mot_src-original_tgt-rain-100_lr1e-4_scratch_noaug"

      num_con=3
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
      num_con=3
      cls_par=0.3
      thr=0
    elif [ "${DATASET}" = "vlcs" ]; then
      MODEL="resnet50"
#      LOG_SUFFIX_SRC="log/cifar10/Src/src_original/tgt_original/220421-1_cifar10_src-original_tgt-original_lr1e-4_pretrained_ep100_uex64_s0/cp/cp_last.pth.tar"
      LOG_SUFFIX_SRC="/home/twkim/git/WWW/log/vlcs/Src/tgt_LabelMe/resnet50_ep100_uex32_s0/cp/cp_last.pth.tar" ### SELECTED


      lr="0.00005"
      num_con=3
      cls_par=0.3
      thr=0

    fi




    if [ "${METHOD}" = "Src" ]; then ################ SRC (pretrain)
      SEED="0"

      if [ "${DATASET}" = "kitti_mot" ] || [ "${DATASET}" = "kitti_sot" ] || [ "${DATASET}" = "vlcs" ]; then
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
--log_suffix $LOG_SUFFIX_SRC \
--update_every_x 64 \
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
--update_every_x 64 \
--log_suffix $LOG_SUFFIX_SRC"
      fi

    elif
      [ "${METHOD}" = "Src_sep" ]
    then ################ SRC (pretrain)
      SEED="0"
      EPOCH=50
      METHOD="Src"
      python script.py \
        --num_concurrent 2 \
        --log_as_file \
        --script \
        "--dataset $DATASET \
\
--src_sep \
\
--model $MODEL \
--method $METHOD \
--epoch $EPOCH \
--seed $SEED \
--log_suffix $LOG_SUFFIX_SRC"

    elif
      [ "${METHOD}" = "FT_all" ]
    then
      LOG_SUFFIX=${DATE}_ftall
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
--update_every_x 200 \
--memory_size 200 \
--memory_type FIFO \
\
--log_suffix ${LOG_SUFFIX}"

    elif
      [ "${METHOD}" = "SHOT" ]
    then

      EPOCH=100
      LOG_SUFFIX=${DATE}_shot
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
--epoch $EPOCH \
--seed $SEED \
--load_checkpoint_suffix $LOG_SUFFIX_SRC \
--tgt_train_dist 0 1 \
--nsample $nsample \
\
--cls_par $cls_par \
--threshold $thr \
\
\
--online \
--epoch 1 \
--update_every_x 200 \
--memory_size 200 \
--memory_type FIFO \
\
\
--log_suffix ${LOG_SUFFIX}"

    elif [ "${METHOD}" = "CDAN" ]; then

      EPOCH=100
      LOG_SUFFIX=${DATE}_cdan
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
--epoch $EPOCH \
--seed $SEED \
--load_checkpoint_suffix $LOG_SUFFIX_SRC \
--tgt_train_dist 0 1 \
--nsample $nsample \
\
--cls_par $cls_par \
--threshold $thr \
\
\
--online \
--epoch 1 \
--update_every_x 200 \
--memory_size 200 \
--memory_type FIFO \
\
\
--log_suffix ${LOG_SUFFIX}"

    elif [ "${METHOD}" = "FeatMatch" ]; then
      LOG_SUFFIX=${DATE}_featmatch
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
--epoch $EPOCH \
--seed $SEED \
--load_checkpoint_suffix $LOG_SUFFIX_SRC \
--tgt_train_dist 0 1 \
--nsample $nsample \
\
--cls_par $cls_par \
--threshold $thr \
\
\
--online \
--epoch 1 \
--update_every_x 200 \
--memory_size 200 \
--memory_type FIFO \
\
\
--log_suffix ${LOG_SUFFIX}"

    elif [ "${METHOD}" = "TT_SINGLE_STATS" ]; then
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
--memory_size 1 \
--online \
\
--bn_momentum ${bn_momentum} \
\
\
--log_suffix ${LOG_SUFFIX}"
    elif [ "${METHOD}" = "TT_BATCH_STATS" ]; then
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
\
--epoch 1 \
--update_every_x ${update_every_x} \
--memory_size ${memory_size} \
\
\
--log_suffix ${LOG_SUFFIX}"
    elif [ "${METHOD}" = "TT_BATCH_STATS_momentum" ]; then


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
--method TT_BATCH_STATS \
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
--use_learned_stats \
--bn_momentum ${bn_momentum} \
\
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
        memory_type="CBFIFO"
        use_learned_stats="--use_learned_stats"
        temperature="1"
        loss_scaler="1"
        iabn="--iabn"

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
\
--log_suffix ${LOG_SUFFIX}"
    elif [ "${METHOD}" = "TT_WHOLE" ]; then
      for batch_size in 16 32 64; do
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
\
--epoch 1 5 \
--update_every_x ${batch_size} \
--memory_size ${batch_size} \
\
\
--log_suffix ${LOG_SUFFIX}"
      done
    elif [ "${METHOD}" = "TT_BATCH_PARAMS" ]; then
      for batch_size in 16 32 64; do
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
\
\
--use_learned_stats \
\
--epoch 1 5 \
--update_every_x ${batch_size} \
--memory_size ${batch_size} \
\
\
--log_suffix ${LOG_SUFFIX}"
      done
    elif [ "${METHOD}" = "VOTE" ]; then
      for batch_size in 16 32 64; do
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
\
--epoch 1 \
--update_every_x ${batch_size} \
--memory_size ${batch_size} \
\
\
--log_suffix ${LOG_SUFFIX}"
      done

    elif [ "${METHOD}" = "TENT_STATS" ]; then
      for batch_size in 16 32 64; do
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
--epoch 1 \
--update_every_x ${batch_size} \
--memory_size ${batch_size} \
--bn_momentum 0.0001 0.01 \
--memory_size 99999 \
\
\
--log_suffix ${LOG_SUFFIX}"
      done

    fi

#    python send_email.py --address $ADDR --title ${SERVER}:WWW@${METHOD}_${DATASET}_${DATE}
  done
done
