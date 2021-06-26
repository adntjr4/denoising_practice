resume="False"
n_thread="8"
while getopts g:t:r opt
do
   case "$opt" in
      g) gpu_id="$OPTARG";;
      t) n_thread="$OPTARG";;
      r) resume="True";;
   esac
done

##### Setting #####

session="self-RBSN_nlf_SIDD"
config="self-RBSN_nlf_SIDD"

###################

cd ..
python train.py --session_name $session --config $config --resume $resume --gpu $gpu_id --thread $n_thread
