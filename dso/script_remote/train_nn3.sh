ratio=0.6
noise=0.1
mode=1
log_dir=./log_remote/noise_${noise}_dataratio_${ratio}
mkdir -p ${log_dir}

job_name=Kdv
echo ${job_name}
log_file=${log_dir}/${job_name}_${mode}.log
python ./dso/NN_remote.py $job_name ${noise} ${ratio} ${mode} ${log_file}

job_name=PDE_divide
echo ${job_name} 
log_file=${log_dir}/${job_name}_${mode}.log
python ./dso/NN_remote.py $job_name ${noise} ${ratio} ${mode} ${log_file}

job_name=Burgers
echo ${job_name}
log_file=${log_dir}/${job_name}_${mode}.log
python ./dso/NN_remote.py $job_name ${noise} ${ratio} ${mode} ${log_file}

job_name=chafee-infante
echo ${job_name}
log_file=${log_dir}/${job_name}_${mode}.log
python ./dso/NN_remote.py $job_name ${noise} ${ratio} ${mode} ${log_file}

job_name=PDE_compound
echo ${job_name}
log_file=${log_dir}/${job_name}_${mode}.log
python ./dso/NN_remote.py $job_name ${noise} ${ratio} ${mode} ${log_file}