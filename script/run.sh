ADDR="taesik.gong@kaist.ac.kr"
SERVER=`hostname`
start_time=$(date +%s)

. script/active/all.sh
#. script/active/all2.sh
#. script/active/src.sh
#. script/active/tmp.sh
#. script/active/ablation.sh
#. script/active/tune.sh
#. script/active/all_sep.sh
#. script/active/all_sep2.sh
#. script/active/kitti_sot_src3.sh
#. script/active/kitti_sot_src2.sh
#. script/active/cifar_src1.sh
#. script/active/cifar_src2.sh
#. script/active/tmp1.sh
#. script/active/tmp2.sh
#. script/active/tmp3.sh
#. script/active/tmp4.sh

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
((sec=elapsed%60, elapsed/=60, min=elapsed%60, hrs=elapsed/60))
timestamp=$(printf "%d:%02d:%02d" $hrs $min $sec)


python send_email.py --address $ADDR --title ${SERVER}:WWW@ALL_DONE:${timestamp}