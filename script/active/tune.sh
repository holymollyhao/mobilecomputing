SERVER=$(hostname)

#  METHODS="Ours TENT TT_SINGLE_STATS TT_BATCH_STATS"

if [ "${SERVER}" = "suzy" ]; then
  DATASETS="cifar10"
  METHODS="Ours"
elif [ "${SERVER}" = "iu" ]; then
  DATASETS="kitti_sot"
  METHODS="Ours"
elif [ "${SERVER}" = "chris" ]; then
  DATASETS="cifar10"
  METHODS="Ours"
elif [ "${SERVER}" = "sooae" ]; then
  DATASETS="harth"
  METHODS="FeatMatch"
fi

#log_suffix="220326_tune"
#log_suffix="220326-2_tune_fullsearch"
#log_suffix="220327_tune"
#log_suffix="220327-2_tune_full"
#log_suffix="220328_tune_tempbugfix"
#log_suffix="220402_no_div"
#log_suffix="220416_kitti"
#log_suffix="220416-2_kitti_bs64"
#log_suffix="220417-1_kitti_rain200"
#log_suffix="220417-2_kitti_rain200_every5"
#log_suffix="220417-3_kitti_rain200__220417-1_kitti_mot_src-2d_obj_tgt-rain-100_lr1e-4_scratch_noaug_ep100_uex64_s0"
#log_suffix="220417-4_kitti_rain200__220417-1_kitti_mot_src-2d_obj_tgt-rain-100_lr1e-4_scratch_noaug_ep100_uex64_s0"
#log_suffix="220419-1_cp-220419-3"
#log_suffix="220419-3_cp-220419-2"
#log_suffix="220419-4_cp-220419-1"
#log_suffix="220419-5_cp-220419-1"
#log_suffix="220419-6_cp-220419-1"
#log_suffix="220419-7_cp-220419-2"
#log_suffix="220419-8_cp-220419-2"
#log_suffix="220419-9_cp-220419-2"
#log_suffix="220419-10_cp-220419-2"
#log_suffix="220419-11_cp-220419-2"
#log_suffix="220421-1_cotta"
#log_suffix="220423-1_tent_dist1_tgt"
#log_suffix="220423-2_tent_dist1_tgt"
#log_suffix="220423-3_tent_dist1_tgt"
#log_suffix="220429_iabn"
log_suffix="220429-1_iabn"

for DATASET in $DATASETS; do
  if [ "${DATASET}" = "kitti_mot" ] || [ "${DATASET}" = "kitti_sot" ]; then
    NUM_GPUS_PER_PROCESS=1
  elif [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar100" ]; then
    NUM_GPUS_PER_PROCESS=8
  else
    NUM_GPUS_PER_PROCESS=4
  fi

  for METHOD in $METHODS; do
    python tune/tune_hyperparams.py --method $METHOD --dataset $DATASET --log_suffix ${log_suffix}_${DATASET}_${METHOD} --num_gpus_per_process $NUM_GPUS_PER_PROCESS
  done
done
