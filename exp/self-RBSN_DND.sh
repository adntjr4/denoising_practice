resume="False"
n_thread="4"
while getopts g:t:r opt
do
   case "$opt" in
      g) gpu_id="$OPTARG";;
      t) n_thread="$OPTARG";;
      r) resume="True";;
   esac
done

##### Setting #####

session="self-RBSN_DND"
config="self-RBSN_DND"

###################

cd ..
# python train.py --session_name $session --config $config --resume $resume --gpu $gpu_id --thread $n_thread
python test.py --session_name $session --config $config --ckpt_epoch 64 --gpu $gpu_id --thread $n_thread