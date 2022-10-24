ratio=1
noise=0.01
mode=0
log_dir=./log_remote/noise_${noise}_dataratio_${ratio}
mkdir -p ${log_dir}

# job_name=Kdv
# echo ${job_name}
# log_file=${log_dir}/${job_name}_${mode}.log
# python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

# job_name=PDE_divide
# echo ${job_name} 
# log_file=${log_dir}/${job_name}_${mode}.log
# python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

# job_name=Burgers
# echo ${job_name}
# log_file=${log_dir}/${job_name}_${mode}.log
# python ./dso/NN.py $job_name ${noise} ${ratio} ${mode} ${log_file}

# job_name=chafee-infante
# echo ${job_name}
# log_file=${log_dir}/${job_name}_${mode}.log
# python ./dso/NN.py $job_name ${noise} ${ratio} ${mode} ${log_file}

job_name=Cahn_Hilliard_2D

echo ${job_name}
log_file=${log_dir}/${job_name}_${mode}.log
CUDA_VISIBLE_DEVICES=1   python -m dso.NN_remote $job_name ${noise} ${ratio} ${mode} ${log_file}