SERVER=$(hostname)

cores=8
LOG_SUFFIX="debug_220506_src"

#officehome pacs vlcs
#Src Src_woBN

if [ "${SERVER}" = "suzy" ]; then
  DATASETS="officehome pacs vlcs"
  METHODS="Src_woBN"
elif [ "${SERVER}" = "iu" ]; then
  DATASETS="pacs"
  METHODS="Src_woBN"
elif [ "${SERVER}" = "chris" ]; then
  DATASETS="officehome pacs vlcs"
  METHODS="Src_woBN"
elif [ "${SERVER}" = "sooae" ]; then
  DATASETS="officehome pacs vlcs"
  METHODS="Src_woBN"
fi

echo SERVER: $SERVER
echo DATASETS: $DATASETS
echo METHODS: $METHODS
#TODO: remove LOG_SUFFIX_SRC. Replace it with a specific pth.
i=0
NUM_MAX_JOB=4
VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  if ((${#background[@]} >= NUM_MAX_JOB)); then
    wait -n
  fi
}

#### T3A src generation

for DATASET in $DATASETS; do
  for METHOD in $METHODS; do
      if [ "${DATASET}" = "vlcs" ]; then
        TGT="Caltech101 LabelMe SUN09 VOC2007"
      elif [ "${DATASET}" = "pacs" ]; then
        TGT="art_painting cartoon photo sketch"
      elif [ "${DATASET}" = "officehome" ]; then
        TGT="Art Clipart RealWorld"
      fi
      MODEL="resnet50_pretrained"
      SEED="0"
      EPOCH=50

      for tgt in $TGT; do
        if  [ "${METHOD}" = "Src" ]; then

          CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python main.py --gpu_idx $((i % 8)) --lr 0.00005 --dataset $DATASET --method Src --tgt "${tgt}" --validation --log_suffix "  " --model $MODEL --epoch $EPOCH --update_every_x 32 --seed $SEED 2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &
          i=$((i + 1))
          wait_n

        elif [ "${METHOD}" = "Src_woBN" ]; then

          CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python main.py --gpu_idx $((i % 8)) --lr 0.00005 --dataset $DATASET --method Src --tgt "${tgt}" --fuse_model --validation --log_suffix "${LOG_SUFFIX}_fused" --model $MODEL --epoch $EPOCH --update_every_x 32 --seed $SEED 2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &
          i=$((i + 1))
          wait_n

        fi
      done
  done
done






#SERVER=$(hostname)
#
#
##DATE="220424-1_baselines"
##DATE="220427-2_iabn"
##DATE="220427-3_iabn_k_1m"
##DATE="220427-4_iabn_k_1m_bugfix"
##DATE="220427-5_ours-reprod"
##DATE="220427-6_ours-cbfifo"
##DATE="220427-7_iabn_k_1b_bugfix"
##DATE="220427-8_iabn_k_1b_bugfix_nograd"
##DATE="220428-8_iabn_k_1e20_nograd"
##DATE="220428-1_iabn_k_1e20_nograd_mu-comment"
##DATE="220428-2_iabn_k_1e20_mu-comment"
##DATE="220428-3_iabn_k_1e20_deepcopy"
##DATE="220428-4_iabn_k_1e20_deepcopy_nograd"
##DATE="220428-5_iabn_k_1e20_deepcopy"
##DATE="220428-6_iabn_k_1e20_deepcopy_nograd"
##DATE="220428-7_iabn_k_3_deepcopy"
##DATE="220428-9_iabn_k_2_deepcopy"
##DATE="220428-10_iabn_k_3_eval_src"
##DATE="220428-11_iabn_k_3_eval_ours"
#
##DATE="220428-12_eval_src"
##DATE="220428-13_ours_no_optim"
##DATE="220428-14_ours_detach_k3"
##DATE="220428-15_ours_detach_k4"
##DATE="220428-16_ours_detach_k1e30"
##DATE="220429-1_cifar100"
##DATE="220428-17_ours_detach_k5"
##DATE="220428-18_ours_detach_k6"
##DATE="220428-19_ours_detach_k10"
#DATE="220504_T3A_src"
#
#
## Datasets : vlcs officehome pacs
## Methods : Src Src_woBN Src_wBN
#
#if [ "${SERVER}" = "suzy" ]; then
#  DATASETS="vlcs"
#  METHODS="Src"
#elif [ "${SERVER}" = "iu" ]; then
#  DATASETS="pacs officehome"
#  METHODS="Src"
#elif [ "${SERVER}" = "chris" ]; then
#  DATASETS="vlcs"
#  METHODS="Src"
#elif [ "${SERVER}" = "sooae" ]; then
#  DATASETS="vlcs"
#  METHODS="Src"
#fi
#
#echo SERVER: $SERVER
#echo DATASETS: $DATASETS
#echo METHODS: $METHODS
#
#for DATASET in $DATASETS; do # extrasensory reallifehar harth hhar wesad ichar icsr ;opportunity gait hhar  #  #icsr  #ichar  # # # # #
#  for METHOD in $METHODS; do #Src FT_all SHOT
#
#    bn_momentum="0.01"
#    update_every_x="64"
#    memory_size="64"
#
#    if  [ "${DATASET}" = "vlcs" ]; then
#      EPOCH=100
#      num_con=3
#      MODEL="resnet50"
#      lr="0.0001"
#      if [[ "$METHOD" == *"woBN"* ]]; then # checks if substring woBN is inside variable METHOD
#        LOG_SUFFIX_SRC="${DATE}_${DATASET}_Src_fuse_lr_${lr}"
#      else
#        LOG_SUFFIX_SRC="${DATE}_${DATASET}_Src_nofuse_lr_${lr}"
#      fi
#    elif  [ "${DATASET}" = "officehome" ]; then
#      EPOCH=100
#      num_con=3
#      MODEL="resnet50"
#      lr="0.0001"
#      if [[ "$METHOD" == *"woBN"* ]]; then # checks if substring woBN is inside variable METHOD
#        LOG_SUFFIX_SRC="${DATE}_${DATASET}_Src_fuse_lr_${lr}"
#      else
#        LOG_SUFFIX_SRC="${DATE}_${DATASET}_Src_nofuse_lr_${lr}"
#      fi
#    elif  [ "${DATASET}" = "pacs" ]; then
#      EPOCH=100
#      num_con=3
#      MODEL="resnet50"
#      lr="0.0001"
#      if [[ "$METHOD" == *"woBN"* ]]; then # checks if substring woBN is inside variable METHOD
#        LOG_SUFFIX_SRC="${DATE}_${DATASET}_Src_fuse_lr_${lr}"
#      else
#        LOG_SUFFIX_SRC="${DATE}_${DATASET}_Src_nofuse_lr_${lr}"
#      fi
#    fi
#
#    if [ "${METHOD}" = "Src" ]; then
#      SEED="0"
#      METHOD="Src"
#      for i in 0 1; do
#        if [ "$i" = "1" ]; then
#          LOG_SUFFIX_SRC="${DATE}_${DATASET}_Src_fuse_lr_${lr}"
#          python script.py \
#          --validation \
#          --num_concurrent 2 \
#          --log_as_file \
#          --script \
#          "--dataset $DATASET \
#--fuse_model \
#--model $MODEL \
#--method $METHOD \
#--epoch $EPOCH \
#--seed $SEED \
#--log_suffix $LOG_SUFFIX_SRC \
#--update_every_x ${update_every_x} \
#--load_checkpoint_suffix resnet50"
#        else
#          LOG_SUFFIX_SRC="${DATE}_${DATASET}_Src_nofuse_lr_${lr}"
#          python script.py \
#          --validation \
#          --num_concurrent 2 \
#          --log_as_file \
#          --script \
#          "--dataset $DATASET \
#--model $MODEL \
#--method $METHOD \
#--epoch $EPOCH \
#--seed $SEED \
#--log_suffix $LOG_SUFFIX_SRC \
#--update_every_x ${update_every_x} \
#--load_checkpoint_suffix resnet50"
#        fi
#      done
#    elif [ "${METHOD}" = "Src_woBN" ]; then
#      EPOCH=100
#      SEED="0"
#      METHOD="Src"
#    python script.py \
#        --validation \
#        --num_concurrent 2 \
#        --log_as_file \
#        --script \
#        "--dataset $DATASET \
#--fuse_model \
#--model $MODEL \
#--method $METHOD \
#--epoch $EPOCH \
#--seed $SEED \
#--log_suffix $LOG_SUFFIX_SRC \
#--update_every_x ${update_every_x} \
#--load_checkpoint_suffix resnet50"
#    elif [ "${METHOD}" = "Src_wBN" ]; then
#      EPOCH=100
#      SEED="0"
#      METHOD="Src"
#    python script.py \
#        --validation \
#        --num_concurrent 2 \
#        --log_as_file \
#        --script \
#        "--dataset $DATASET \
#--model $MODEL \
#--method $METHOD \
#--epoch $EPOCH \
#--seed $SEED \
#--log_suffix $LOG_SUFFIX_SRC \
#--update_every_x ${update_every_x} \
#--load_checkpoint_suffix resnet50"
#    fi
#  done
#done
