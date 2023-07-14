ratio_list="1.0"
for ratio in $ratio_list
do
noise=0.01
mode=0
log_dir=./log_local/noise_${noise}_dataratio_${ratio}
mkdir -p ${log_dir}

# job_name=Kdv
# echo ${job_name}
# log_file=${log_dir}/${job_name}_${mode}.log
# python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

job_name=PDE_divide
echo ${job_name} 
log_file=${log_dir}/${job_name}_${mode}_new.log
python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

done
# job_name=Burgers
# echo ${job_name}
# log_file=${log_dir}/${job_name}_${mode}.log
# python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

# job_name=chafee-infante
# echo ${job_name}
# log_file=${log_dir}/${job_name}_${mode}.log
# python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

# job_name=PDE_compound
# echo ${job_name}
# log_file=${log_dir}/${job_name}_${mode}.log
# python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}
# done