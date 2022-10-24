

# # job_name=Kdv
# # echo ${job_name}
# # log_file=${log_dir}/${job_name}_${mode}.log
# # python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

# # job_name=PDE_divide
# # echo ${job_name} 
# # log_file=${log_dir}/${job_name}_${mode}.log
# # python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

# # job_name=Burgers
# # echo ${job_name}
# # log_file=${log_dir}/${job_name}_${mode}.log
# # python ./dso/NN.py $job_name ${noise} ${ratio} ${mode} ${log_file}

# # job_name=chafee-infante
# # echo ${job_name}
# # log_file=${log_dir}/${job_name}_${mode}.log
# # python ./dso/NN.py $job_name ${noise} ${ratio} ${mode} ${log_file}

 
ratio_list="1.0"
noise_list="0.1"
datasets="chafee-infante"
# datasets="PDE_compound PDE_divide"
mode="train"
# PDE_compound"
# datasets="chafee-infante"
for dataset in ${datasets}
do
  for ratio in ${ratio_list}
  do
    for noise in ${noise_list}
    do
    log_dir=./log_NN/noise_${noise}_dataratio_${ratio}
    mkdir -p ${log_dir}
    job_name=$dataset
    echo ${job_name}_${log_dir}


    normalize=None
    log_file=${log_dir}/${job_name}_${normalize}.log
    python train_NN.py --data-name $job_name --noise-level ${noise}  --train-ratio ${ratio} --normalize-type ${normalize}  \
    --num-layer 5 --hidden-num 50 --maxit 5000 --early-stopper 10000 --activation tanh \
    --mode ${mode} > ${log_file}_${mode} 2>&1
    done
  done
done