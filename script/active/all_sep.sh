SERVER=$(hostname)

#DATE="220327_tuned"
DATE="220327-0_debug"

#harth reallifehar extrasensory
#Src FT_all CDAN SHOT FeatMatch

if [ "${SERVER}" = "suzy" ]; then
  DATASETS="extrasensory"
  METHODS="TT_BATCH_PARAMS"
elif [ "${SERVER}" = "iu" ]; then
  DATASETS="harth"
  METHODS="Src"
elif [ "${SERVER}" = "chris" ]; then
  DATASETS="reallifehar extrasensory"
  METHODS="Src"
elif [ "${SERVER}" = "sooae" ]; then
  DATASETS="harth"
  METHODS="FeatMatch"
fi

echo SERVER: $SERVER
echo DATASETS: $DATASETS
echo METHODS: $METHODS

for DATASET in $DATASETS; do # extrasensory reallifehar harth hhar wesad ichar icsr ;opportunity gait hhar  #  #icsr  #ichar  # # # # #
  for METHOD in $METHODS; do #Src FT_all SHOT

    if [ "${DATASET}" = "harth" ]; then
      MODEL="HARTH_model"
#      LOG_SUFFIX_SRC="220327-1_harth_minmax_scaling_all_split_win50"       # src-back, tgt-thigh ########### SELECTED
      LOG_SUFFIX_SRC="220327-2_harth_minmax_scaling_all_split_win50_val" # src-back, tgt-thigh ########### SELECTED
      num_con=3
      cls_par=0.3
      thr=0

    elif [ "${DATASET}" = "extrasensory" ]; then
      MODEL="ExtraSensory_model"

#      LOG_SUFFIX_SRC="220327-1_extrasensory_selectedfeat_woutloc_std_scaling_all_win5"       ############ SELECTED
      LOG_SUFFIX_SRC="220327-2_extrasensory_selectedfeat_woutloc_std_scaling_all_win5_val" ############ SELECTED
      num_con=3
      cls_par=0.3
      thr=0
    elif [ "${DATASET}" = "reallifehar" ]; then
      MODEL="RealLifeHAR_model"

#      LOG_SUFFIX_SRC="220327-1_reallifehar_acc_minmax_scaling_all_win400_overlap0"       ############ SELECTED
      LOG_SUFFIX_SRC="220327-2_reallifehar_acc_minmax_scaling_all_win400_overlap0_val" ############ SELECTED

      num_con=3
      cls_par=0.3
      thr=0

    fi

    if [ "${METHOD}" = "Src" ]; then ################ SRC (pretrain)
      SEED="0"
      EPOCH=50

      python script.py \
        --num_concurrent 2 \
        --log_as_file \
        --validation \
        --script \
        "--dataset $DATASET \
--model $MODEL \
--method $METHOD \
--epoch $EPOCH \
--seed $SEED \
--log_suffix $LOG_SUFFIX_SRC"

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
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
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
--epoch $EPOCH \
--remove_cp \
--seed $SEED \
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
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
--epoch $EPOCH \
--remove_cp \
--seed $SEED \
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
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
--epoch $EPOCH \
--remove_cp \
--seed $SEED \
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
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
      if [ "${DATASET}" = "harth" ]; then
        bn_momentum="0.0001"
      elif [ "${DATASET}" = "extrasensory" ]; then
        bn_momentum="0.001"
      elif [ "${DATASET}" = "reallifehar" ]; then
        bn_momentum="0.0001"
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
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
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
      for batch_size in 64; do
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
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
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

    elif [ "${METHOD}" = "TENT" ]; then
      if [ "${DATASET}" = "harth" ]; then
        lr="0.0001"
        bn_momentum="0.001"
        update_every_x="64"
        memory_size="64"
        use_learned_stats="--use_learned_stats"
      elif [ "${DATASET}" = "extrasensory" ]; then
        lr="0.0001"
        bn_momentum="0.1"
        update_every_x="64"
        memory_size="64"
        use_learned_stats="--use_learned_stats"
      elif [ "${DATASET}" = "reallifehar" ]; then
        lr="0.0001"
        bn_momentum="0.0001"
        update_every_x="64"
        memory_size="64"
        use_learned_stats="--use_learned_stats"
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
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
--tgt_train_dist 0 1 \
--nsample $nsample \
--online \
\
--lr ${lr} \
--bn_momentum ${bn_momentum} \
--update_every_x ${update_every_x} \
--memory_size ${memory_size} \
${use_learned_stats} \
\
\
--log_suffix ${LOG_SUFFIX}"

    elif [ "${METHOD}" = "Ours" ]; then
      if [ "${DATASET}" = "harth" ]; then
        lr="0.0001"
        temperature="0.1"
        loss_scaler="0.5"
        bn_momentum="0.01"
        update_every_x="64"
        memory_size="64"
        memory_type="CBFIFO"
        use_learned_stats="--use_learned_stats"
      elif [ "${DATASET}" = "extrasensory" ]; then

        lr="0.001"
        temperature="20"
        loss_scaler="20"
        bn_momentum="0.1"
        update_every_x="64"
        memory_size="64"
        memory_type="CBFIFO"
        use_learned_stats="--use_learned_stats"
      elif [ "${DATASET}" = "reallifehar" ]; then

        lr="0.00001"
        temperature="10"
        loss_scaler="10"
        bn_momentum="0.0001"
        update_every_x="64"
        memory_size="64"
        memory_type="CBFIFO"
        use_learned_stats="--use_learned_stats"
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
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
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
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
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
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
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
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
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
--load_checkpoint_suffix ${LOG_SUFFIX_SRC}_ep50_s0 \
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

    python send_email.py --address $ADDR --title ${SERVER}:WWW@${METHOD}_${DATASET}_${DATE}
  done
done
