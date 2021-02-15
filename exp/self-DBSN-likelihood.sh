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

session="self-DBSN-likelihood"
config="self-DBSN-likelihood"

###################

cd ..
if [ $resume == "True" ]
then
  python train.py --session_name $session \
                  --config $config \
                  --resume \
                  --gpu $gpu_id \
                  --thread $n_thread
else
  python train.py --session_name $session \
                  --config $config \
                  --gpu $gpu_id \
                  --thread $n_thread
fi
