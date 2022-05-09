SERVER=$(hostname)
#TODO: Log_suffix -> log_prefix
#LOG_SUFFIX="220504_src_ep50"
#LOG_SUFFIX="220505-1_src_eval"
#LOG_SUFFIX="220505-4_ours_eval_optim"
#LOG_SUFFIX="220505-5_ours_eval_no-optim"
#LOG_SUFFIX="220505-debug"
#LOG_SUFFIX="220505-6_ours_eval_optim"
#LOG_SUFFIX="220505-7_ours_eval_no-optim"
#LOG_SUFFIX="220505-8_bn"
#LOG_SUFFIX="220505_src_ep50"
#LOG_SUFFIX="220505-9_eval"
#LOG_SUFFIX="220505-2_src_ep50_r18_pretrained"
#LOG_SUFFIX="220505-3_src_ep100_r18_scratch"
#LOG_SUFFIX="220505-10_src_ep50_r18_pretrained-b128"
#LOG_SUFFIX="220505-10_src_ep50_r18_pretrained-b256"
#LOG_SUFFIX="220505-11_src_ep50_r18_pretrained-test"
#LOG_SUFFIX="220505_src_ep100_r18_scratch_coslr"
#LOG_SUFFIX="220505_src_ep100_coslr"

#LOG_SUFFIX="220506_src_ep100_coslr"
#LOG_SUFFIX="220505_src_ep200_coslr"
#LOG_SUFFIX="220506_ours-no_optim"

#LOG_SUFFIX="220506_src_ep200_coslr"
#LOG_SUFFIX="220506_src_2d_object"
#LOG_SUFFIX="220506_src_ep200_coslr_norm"

#LOG_SUFFIX="220506_src_ep200_224"
#LOG_SUFFIX="220506-1_src_2d_object_eval_baseline_ours-no_optim"
#LOG_SUFFIX="220507_src_ep200_coslr"
#LOG_SUFFIX="debug"
#LOG_SUFFIX="220506_src_ep200_224"
#LOG_SUFFIX="220507_src_ep200_wres2810_coslr"

#LOG_SUFFIX="220507_ablation"
#LOG_SUFFIX="220507_recheck"
#LOG_SUFFIX="220507-2_eval"
#LOG_SUFFIX="220507-3_eval_t3a_cotta"
#LOG_SUFFIX="220507-4_cotta_debug"
#LOG_SUFFIX="220507-5_cotta_debug_wideresnet28"
#LOG_SUFFIX="220507-6_cotta_debug_every200"
#LOG_SUFFIX="220507_src_DG_data_ep50"
#LOG_SUFFIX="220507-7_cotta_debug_every200"
#LOG_SUFFIX="220508_vcls_debug"
#LOG_SUFFIX="220508_pacs_debug"

#LOG_SUFFIX="220508_ablation_k4"
#LOG_SUFFIX="220508_baselines"
#LOG_SUFFIX="220508-1_baselines"
#LOG_SUFFIX="220508-2_src_eval"
#LOG_SUFFIX="220508_baselines"

LOG_SUFFIX="220508-3_cotta"

#cifar10 cifar100 harth reallifehar extrasensory kitti_sot vlcs pacs
#Src TT_BATCH_STATS PseudoLabel TENT T3A COTTA Ours Src_woBN
#harth reallifehar extrasensory

if [ "${SERVER}" = "suzy" ]; then
  DATASETS="cifar10"
  METHODS="COTTA"
elif [ "${SERVER}" = "iu" ]; then
  DATASETS="cifar100"
  METHODS="COTTA"
elif [ "${SERVER}" = "chris" ]; then
  DATASETS="vlcs pacs"
  METHODS="Src"
elif [ "${SERVER}" = "sooae" ]; then
  DATASETS="vlcs"
  METHODS="Ours"
fi

#TODO
#Run T3A
#Run VLCS,PACS
#Run Ablation study (k=4, memory-centered)
#Run COTTA for vcls
#Run Ours for pacs

echo SERVER: $SERVER
echo DATASETS: $DATASETS
echo METHODS: $METHODS

NUM_MAX_JOB=8
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

#### Useful functions
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  if ((${#background[@]} >= NUM_MAX_JOB)); then
    wait -n
  fi
}

###############################################################
##### Source Training; Source Evaluation: Source domains  #####
###############################################################
train_source_model() {
  i=0
  update_every_x="64"
  memory_size="64"
  for DATASET in $DATASETS; do # extrasensory reallifehar harth hhar wesad ichar icsr ;opportunity gait hhar  #  #icsr  #ichar  # # # # #
    for METHOD in $METHODS; do #Src FT_all SHOT

      validation="--dummy"

      if [ "${DATASET}" = "harth" ]; then
        EPOCH=100
        MODEL="HARTH_model"
        TGT="src"
      elif [ "${DATASET}" = "extrasensory" ]; then
        EPOCH=100
        MODEL="ExtraSensory_model"
        TGT="src"
      elif [ "${DATASET}" = "reallifehar" ]; then
        EPOCH=100
        MODEL="RealLifeHAR_model"
        TGT="src"
      elif [ "${DATASET}" = "kitti_sot" ]; then
        EPOCH=50
        MODEL="resnet50_pretrained"
        TGT="src"
      elif [ "${DATASET}" = "cifar10" ]; then
        EPOCH=200
        MODEL="resnet18"
        #        MODEL="wideresnet28-10"
        TGT="test"
      elif [ "${DATASET}" = "cifar100" ]; then
        EPOCH=200
        MODEL="resnet18"
        #        MODEL="wideresnet28-10"
        TGT="test"
      elif [ "${DATASET}" = "vlcs" ]; then
        EPOCH=50
        MODEL="resnet50_pretrained"
        TGT="Caltech101 LabelMe SUN09 VOC2007"
        validation="--validation"
      elif [ "${DATASET}" = "pacs" ]; then
        EPOCH=50
        MODEL="resnet50_pretrained"
        TGT="art_painting cartoon photo sketch"
        validation="--validation"
      fi

      SEED="0"
      if [[ "$METHOD" == *"Src"* ]]; then
        #### Train with BN
        for tgt in $TGT; do
          if [ "$METHOD" = "Src_woBN" ]; then #Training source model by fusing BN layers
            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method Src --tgt ${tgt} --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} --memory_size ${memory_size} --seed $SEED \
              ${validation} \
              --fuse_model \
              --log_suffix ${LOG_SUFFIX}_fused \
              2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

            i=$((i + 1))
            wait_n
          else #Normal training
            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method Src --tgt ${tgt} --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} --memory_size ${memory_size} --seed $SEED \
              --log_suffix ${LOG_SUFFIX} \
              ${validation} \
              2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

            i=$((i + 1))
            wait_n

            ### Train with IABN, no fuse
            for iabn_k in 3 4 5; do
              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method Src --tgt ${tgt} --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} --memory_size ${memory_size} --seed $SEED \
                --iabn --iabn_k ${iabn_k} \
                --log_suffix ${LOG_SUFFIX}_iabn_k${iabn_k} \
                ${validation} \
                2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done
          fi
        done
      fi
    done
  done

  wait
}

test_time_adaptation() {
  ###############################################################
  ###### Run Baselines & Ours; Evaluation: Target domains  ######
  ###############################################################

  i=0

  for DATASET in $DATASETS; do # extrasensory reallifehar harth hhar wesad ichar icsr ;opportunity gait hhar  #  #icsr  #ichar  # # # # #
    for METHOD in $METHODS; do #Src FT_all SHOT

      bn_momentum="0.01"
      update_every_x="64"
      memory_size="64"
      SEED="0"
      lr="0.001"
      validation="--dummy"
      weight_decay="0"
      if [ "${DATASET}" = "harth" ]; then
        MODEL="HARTH_model"
        #      CP_base="log/harth/Src/tgt_src/220504_src_ep50"
        CP_base="log/harth/Src/tgt_src/220505_src_ep100_coslr"

        TGTS="S008_thigh
            S018_thigh
            S019_thigh
            S021_thigh
            S022_thigh
            S028_thigh
            S029_thigh"

      elif [ "${DATASET}" = "extrasensory" ]; then
        MODEL="ExtraSensory_model"
        #      CP_base="log/extrasensory/Src/tgt_src/220504_src_ep50"
        CP_base="log/extrasensory/Src/tgt_src/220505_src_ep100_coslr"

        TGTS="4FC32141-E888-4BFF-8804-12559A491D8C
            59818CD2-24D7-4D32-B133-24C2FE3801E5
            61976C24-1C50-4355-9C49-AAE44A7D09F6
            797D145F-3858-4A7F-A7C2-A4EB721E133C
            A5CDF89D-02A2-4EC1-89F8-F534FDABDD96
            C48CE857-A0DD-4DDB-BEA5-3A25449B2153
            D7D20E2E-FC78-405D-B346-DBD3FD8FC92B"

      elif
        [ "${DATASET}" = "reallifehar" ]
      then

        MODEL="RealLifeHAR_model"
        #      CP_base="log/reallifehar/Src/tgt_src/220504_src_ep50"
        CP_base="log/reallifehar/Src/tgt_src/220505_src_ep100_coslr"

        TGTS="p12 p2 p6 p7 p9"

      elif [ "${DATASET}" = "kitti_sot" ]; then

        MODEL="resnet50_pretrained"
        #      CP_base="log/kitti_sot/Src/tgt_src/220504_src_ep50"
        CP_base="log/kitti_sot/Src/tgt_src/220506_src_2d_object"
        #      MODEL="resnet34_pretrained"
        #      CP_base="log/kitti_sot/Src/tgt_src/220505_src_ep50"

        TGTS="rain-200"

      elif [ "${DATASET}" = "cifar10" ]; then
        MODEL="resnet18"
        CP_base="log/cifar10/Src/tgt_test/220507_src_ep200_coslr"
        #        MODEL="wideresnet28-10"
        #        CP_base="log/cifar10/Src/tgt_test/220507_src_ep200_wres2810_coslr"
        #              TGTS="test"
        TGTS="gaussian_noise-5
            shot_noise-5
            impulse_noise-5
            defocus_blur-5
            glass_blur-5
            motion_blur-5
            zoom_blur-5
            snow-5
            frost-5
            fog-5
            brightness-5
            contrast-5
            elastic_transform-5
            pixelate-5
            jpeg_compression-5"

      elif [ "${DATASET}" = "cifar100" ]; then
        MODEL="resnet18"
        CP_base="log/cifar100/Src/tgt_test/220507_src_ep200_coslr"
        #        MODEL="wideresnet28-10"
        #        CP_base="log/cifar100/Src/tgt_test/220507_src_ep200_wres2810_coslr"
        #              TGTS="test"
        TGTS="gaussian_noise-5
            shot_noise-5
            impulse_noise-5
            defocus_blur-5
            glass_blur-5
            motion_blur-5
            zoom_blur-5
            snow-5
            frost-5
            fog-5
            brightness-5
            contrast-5
            elastic_transform-5
            pixelate-5
            jpeg_compression-5"

      elif [ "${DATASET}" = "vlcs" ]; then
        CP_base="220507_src_DG_data_ep50"
        validation='--validation'
        MODEL="resnet50_pretrained"
        TGTS="Caltech101
            VOC2007
            LabelMe
            SUN09"
      elif [ "${DATASET}" = "pacs" ]; then
        CP_base="220507_src_DG_data_ep50"
        validation='--validation'
        MODEL="resnet50_pretrained"
        TGTS="cartoon
            photo
            art_painting
            sketch"
      fi

      if [ "${METHOD}" = "Src" ]; then
        EPOCH=0
        #### Train with BN
        CP=${CP_base}/cp/cp_last.pth.tar
        for TGT in $TGTS; do
          fuse_model="--dummy"
          if [ "${DATASET}" = "vlcs" ] || [ "${DATASET}" = "pacs" ]; then

            CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}/cp/cp_last.pth.tar

            ####If want to eval fused model
            #            CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}_fused/cp/cp_last_fused.pth.tar
            #            fuse_model="--fuse_model"
          fi

          python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} --load_checkpoint_path ${CP} --seed $SEED \
            --log_suffix ${LOG_SUFFIX} \
            $validation \
            $fuse_model \
            2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

          i=$((i + 1))
          wait_n
        done

        #### Model with IABN
        for iabn_k in 3 4 5; do
          CP=${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar

          for TGT in $TGTS; do
            if [ "${DATASET}" = "vlcs" ] || [ "${DATASET}" = "pacs" ]; then
              CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar
            fi
            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} --load_checkpoint_path ${CP} --seed $SEED \
              --iabn --iabn_k ${iabn_k} \
              --log_suffix ${LOG_SUFFIX}_iabn_k${iabn_k} \
              $validation \
              2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

            i=$((i + 1))
            wait_n
          done
        done

      elif [ "${METHOD}" = "Ours" ]; then
        EPOCH=1
        memory_type="CBReservoir"
        #      no_optim="--dummy"
        no_optim="--no_optim"

        #### Train with BN

        for dist in 0 1; do
          for bn_momentum in 0.1 0.05 0.01 0.001; do
            CP=${CP_base}/cp/cp_last.pth.tar
            for TGT in $TGTS; do
              if [ "${DATASET}" = "vlcs" ] || [ "${DATASET}" = "pacs" ]; then
                CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}/cp/cp_last.pth.tar
              fi
              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
                --remove_cp --online --use_learned_stats --lr ${lr} --weight_decay ${weight_decay} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type} --bn_momentum ${bn_momentum} \
                ${no_optim} \
                --log_suffix ${LOG_SUFFIX}_dist${dist}_mt${bn_momentum} \
                $validation \
                2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done

            ### Train with IABN
            for iabn_k in 3 4 5; do
              CP=${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar

              for TGT in $TGTS; do
                if [ "${DATASET}" = "vlcs" ] || [ "${DATASET}" = "pacs" ]; then
                  CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar
                fi

                python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --log_suffix ${LOG_SUFFIX}_iabn_k${iabn_k} --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
                  --remove_cp --online --use_learned_stats --lr ${lr} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type} --bn_momentum ${bn_momentum} \
                  --iabn --iabn_k ${iabn_k} \
                  ${no_optim} \
                  --log_suffix ${LOG_SUFFIX}_dist${dist}_iabn_k${iabn_k}_mt${bn_momentum} \
                  $validation \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          done
        done
      elif [ "${METHOD}" = "TT_BATCH_STATS" ]; then

        #### Train with BN
        for dist in 0 1; do
          CP=${CP_base}/cp/cp_last.pth.tar
          for TGT in $TGTS; do
            if [ "${DATASET}" = "vlcs" ] || [ "${DATASET}" = "pacs" ]; then
              CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}/cp/cp_last.pth.tar
            fi

            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
              --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
              --log_suffix ${LOG_SUFFIX}_dist${dist} \
              $validation \
              2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

            i=$((i + 1))
            wait_n
          done

          #        for iabn_k in 3 4 5; do
          #          CP=${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar
          #
          #          for TGT in $TGTS; do
          #            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
          #              --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
          #              --iabn --iabn_k ${iabn_k} \
          #              --log_suffix ${LOG_SUFFIX}_dist${dist}_iabn_k${iabn_k} \
          #              2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &
          #
          #            i=$((i + 1))
          #            wait_n
          #          done
          #        done

        done
      elif [ "${METHOD}" = "PseudoLabel" ]; then
        EPOCH=1
        #### Train with BN
        for dist in 0 1; do
          CP=${CP_base}/cp/cp_last.pth.tar
          for TGT in $TGTS; do
            if [ "${DATASET}" = "vlcs" ] || [ "${DATASET}" = "pacs" ]; then
              CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}/cp/cp_last.pth.tar
            fi

            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
              --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
              --lr ${lr} --weight_decay ${weight_decay} \
              --log_suffix ${LOG_SUFFIX}_dist${dist} \
              $validation \
              2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

            i=$((i + 1))
            wait_n
          done
        done
      elif [ "${METHOD}" = "TENT" ]; then
        EPOCH=1
        #### Train with BN
        for dist in 0 1; do
          CP=${CP_base}/cp/cp_last.pth.tar
          for TGT in $TGTS; do
            if [ "${DATASET}" = "vlcs" ] || [ "${DATASET}" = "pacs" ]; then
              CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}/cp/cp_last.pth.tar
            fi

            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
              --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
              --lr ${lr} --weight_decay ${weight_decay} \
              --log_suffix ${LOG_SUFFIX}_dist${dist} \
              $validation \
              2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

            i=$((i + 1))
            wait_n
          done
        done

      elif [ "${METHOD}" = "T3A" ]; then
        EPOCH=1
        #### Train with BN
        for dist in 0 1; do
          CP=${CP_base}/cp/cp_last.pth.tar
          for TGT in $TGTS; do

            if [ "${DATASET}" = "vlcs" ] || [ "${DATASET}" = "pacs" ]; then
              CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}_fused/cp/cp_last_fused.pth.tar #For T3A with VLCS and PACS, use model without BN
            fi

            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method T3A --tgt ${TGT} --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
              --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
              --log_suffix ${LOG_SUFFIX}_dist${dist} \
              --fuse_model \
              $validation \
              2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

            i=$((i + 1))
            wait_n
          done
        done
      elif [ "${METHOD}" = "COTTA" ]; then
        EPOCH=1
        #        update_every_x=200
        #        memory_size=200

        if [ "${DATASET}" = "harth" ]; then
          aug_threshold=0 #0.46, no aug for sensor data
        elif [ "${DATASET}" = "extrasensory" ]; then
          aug_threshold=0 #0.51, no aug for sensor data
        elif [ "${DATASET}" = "reallifehar" ]; then
          aug_threshold=0 #0.52, no aug for sensor data
        elif [ "${DATASET}" = "kitti_sot" ]; then
          aug_threshold=0.55
        elif [ "${DATASET}" = "cifar10" ]; then
          aug_threshold=0.92 #value reported from the official code
          #          aug_threshold=0
        elif [ "${DATASET}" = "cifar100" ]; then
          aug_threshold=0.72 #value reported from the official code
          #          aug_threshold=0
        elif [ "${DATASET}" = "vlcs" ]; then
          # tgt_caltech
          #          aug_threshold=0.94
          #          # tgt_labelme
          #          aug_threshold=0.95
          #          # tgt_sun
          #          aug_threshold=0.95
          # tgt_voc
          aug_threshold=0.95
        elif [ "${DATASET}" = "pacs" ]; then
          # art_painting
          #          aug_threshold=0.95
          #          # catoon
          #          aug_threshold=0.95
          #          # photo
          #          aug_threshold=0.95
          # sketch
          aug_threshold=0.95
        fi

        #### Train with BN
        for dist in 0 1; do
          CP=${CP_base}/cp/cp_last.pth.tar
          for TGT in $TGTS; do
            if [ "${DATASET}" = "vlcs" ] || [ "${DATASET}" = "pacs" ]; then
              CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}/cp/cp_last.pth.tar
            fi

            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
              --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
              --lr ${lr} --weight_decay ${weight_decay} \
              --aug_threshold ${aug_threshold} \
              --log_suffix ${LOG_SUFFIX}_dist${dist} \
              $validation \
              2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_job${i}.txt &

            i=$((i + 1))
            wait_n
          done
        done
      fi

    done
  done

  wait
}

ablation() {
  ###############################################################
  ################ Ablation study of our method  ################
  ###############################################################

  i=0

  for DATASET in $DATASETS; do # extrasensory reallifehar harth hhar wesad ichar icsr ;opportunity gait hhar  #  #icsr  #ichar  # # # # #
    for METHOD in $METHODS; do #Src FT_all SHOT

      bn_momentum="0.01"
      update_every_x="64"
      memory_size="64"
      SEED="0"
      lr="0.001"
      weight_decay="0"
      validation="--dummy"
      if [ "${DATASET}" = "harth" ]; then
        MODEL="HARTH_model"
        #      CP_base="log/harth/Src/tgt_src/220504_src_ep50"
        CP_base="log/harth/Src/tgt_src/220505_src_ep100_coslr"

        TGTS="S008_thigh
            S018_thigh
            S019_thigh
            S021_thigh
            S022_thigh
            S028_thigh
            S029_thigh"

      elif [ "${DATASET}" = "extrasensory" ]; then
        MODEL="ExtraSensory_model"
        CP_base="log/extrasensory/Src/tgt_src/220505_src_ep100_coslr"

        TGTS="4FC32141-E888-4BFF-8804-12559A491D8C
            59818CD2-24D7-4D32-B133-24C2FE3801E5
            61976C24-1C50-4355-9C49-AAE44A7D09F6
            797D145F-3858-4A7F-A7C2-A4EB721E133C
            A5CDF89D-02A2-4EC1-89F8-F534FDABDD96
            C48CE857-A0DD-4DDB-BEA5-3A25449B2153
            D7D20E2E-FC78-405D-B346-DBD3FD8FC92B"

      elif [ "${DATASET}" = "reallifehar" ]; then
        MODEL="RealLifeHAR_model"
        CP_base="log/reallifehar/Src/tgt_src/220505_src_ep100_coslr"

        TGTS="p12 p2 p6 p7 p9"

      elif [ "${DATASET}" = "kitti_sot" ]; then
        MODEL="resnet50_pretrained"
        CP_base="log/kitti_sot/Src/tgt_src/220506_src_2d_object"

        TGTS="rain-200"

      elif [ "${DATASET}" = "cifar10" ]; then

        MODEL="resnet18"
        CP_base="log/cifar10/Src/tgt_test/220507_src_ep200_coslr"
        #        MODEL="wideresnet28-10"
        #        CP_base="log/cifar10/Src/tgt_test/220507_src_ep200_wres2810_coslr"
        #      TGTS="test"
        TGTS="gaussian_noise-5
            shot_noise-5
            impulse_noise-5
            defocus_blur-5
            glass_blur-5
            motion_blur-5
            zoom_blur-5
            snow-5
            frost-5
            fog-5
            brightness-5
            contrast-5
            elastic_transform-5
            pixelate-5
            jpeg_compression-5"

      elif [ "${DATASET}" = "cifar100" ]; then
        MODEL="resnet18"
        CP_base="log/cifar100/Src/tgt_test/220507_src_ep200_coslr"
        #        MODEL="wideresnet28-10"
        #        CP_base="log/cifar100/Src/tgt_test/220507_src_ep200_wres2810_coslr"
        #      TGTS="test"
        TGTS="gaussian_noise-5
            shot_noise-5
            impulse_noise-5
            defocus_blur-5
            glass_blur-5
            motion_blur-5
            zoom_blur-5
            snow-5
            frost-5
            fog-5
            brightness-5
            contrast-5
            elastic_transform-5
            pixelate-5
            jpeg_compression-5"

      elif [ "${DATASET}" = "vlcs" ]; then
        CP_base="220507_src_DG_data_ep50"
        validation='--validation'
        MODEL="resnet50_pretrained"
        TGTS="Caltech101
            VOC2007
            LabelMe
            SUN09"
      elif [ "${DATASET}" = "pacs" ]; then
        CP_base="220507_src_DG_data_ep50"
        validation='--validation'
        MODEL="resnet50_pretrained"
        TGTS="cartoon
            photo
            art_painting
            sketch"
      fi

      #### Train with BN
      #case1 case2 case3 case4 case5 case6 case7 case8
      #case2 case3 case4 case5 case6 case9 case10 case11
      for ablation in case2 case3 case4 case5 case6 case9 case10 case11; do
        for bn_momentum in 0.01; do
          for iabn_k in 4; do #3 4 5
            METHOD="Ours"
            EPOCH=1
            memory_type="FIFO"
            no_optim="--no_optim"
            iabn="--dummy"
            loss_scaler="1"
            online="--online"
            CP=${CP_base}/cp/cp_last.pth.tar
            if [ "${ablation}" = "case1" ]; then
              METHOD="Src"
              EPOCH=0
              iabn="--iabn"
              online="--dummy"
              CP=${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar
            elif [ "${ablation}" = "case2" ]; then
              :
            elif [ "${ablation}" = "case3" ]; then
              memory_type="Reservoir"
            elif [ "${ablation}" = "case4" ]; then
              memory_type="CBFIFO"
            elif [ "${ablation}" = "case5" ]; then
              memory_type="CBReservoir"
            elif [ "${ablation}" = "case6" ]; then # current method
              memory_type="CBReservoir"
              iabn="--iabn"
              CP=${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar
            elif [ "${ablation}" = "case7" ]; then
              no_optim="--dummy"
              loss_scaler="0"
              memory_type="CBReservoir"
              iabn="--iabn"
              CP=${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar
            elif [ "${ablation}" = "case8" ]; then
              no_optim="--dummy"
              memory_type="CBReservoir"
              iabn="--iabn"
              CP=${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar
            elif [ "${ablation}" = "case9" ]; then
              memory_type="FIFO"
              iabn="--iabn"
              CP=${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar
            elif [ "${ablation}" = "case10" ]; then
              memory_type="Reservoir"
              iabn="--iabn"
              CP=${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar
            elif [ "${ablation}" = "case11" ]; then
              memory_type="CBFIFO"
              iabn="--iabn"
              CP=${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar
            elif [ "${ablation}" = "casex" ]; then ######################## update_all
              update_all="--update_all"            #TODO
            fi

            for dist in 0 1; do
              for TGT in $TGTS; do
                if [ "${DATASET}" = "vlcs" ] || [ "${DATASET}" = "pacs" ]; then
                  if [ "${iabn}" = "--iabn" ]; then
                    CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}_iabn_k${iabn_k}/cp/cp_last.pth.tar
                  else
                    CP=log/${DATASET}/Src/tgt_${TGT}/${CP_base}/cp/cp_last.pth.tar
                  fi
                fi
                python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
                  --remove_cp --use_learned_stats --lr ${lr} --weight_decay ${weight_decay} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --bn_momentum ${bn_momentum} \
                  ${no_optim} --loss_scaler ${loss_scaler} ${online} \
                  ${iabn} --iabn_k ${iabn_k} \
                  --memory_type ${memory_type} \
                  ${validation} \
                  --log_suffix ${LOG_SUFFIX}_dist${dist}_mt${bn_momentum}_k${iabn_k}_${ablation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_SUFFIX}_dist${dist}_mt${bn_momentum}_k${iabn_k}_${ablation}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          done
        done
      done

    done
  done

  wait
}

#train_source_model
test_time_adaptation
#ablation
