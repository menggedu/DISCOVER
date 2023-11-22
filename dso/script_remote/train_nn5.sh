ratio=1
noise=0.01
mode=1
log_dir=./log_remote/noise_${noise}_dataratio_${ratio}
mkdir -p ${log_dir}

# job_name=Kdv
# echo ${job_name}
# log_file=${log_dir}/${job_name}_${mode}.log
# python ./dso/NN_remote.py $job_name ${noise} ${ratio} ${mode} ${log_file}

# job_name=PDE_divide
# echo ${job_name} 
# log_file=${log_dir}/${job_name}_${mode}.log
# python ./dso/NN_remote.py $job_name ${noise} ${ratio} ${mode} ${log_file}

# job_name=Burgers
# echo ${job_name}
# log_file=${log_dir}/${job_name}_${mode}.log
# python ./dso/NN_remote.py $job_name ${noise} ${ratio} ${mode} ${log_file}

# job_name=chafee-infante
# echo ${job_name}
# log_file=${log_dir}/${job_name}_${mode}.log
# python ./dso/NN_remote.py $job_name ${noise} ${ratio} ${mode} ${log_file}

job_name=Cahn_Hilliard_2D
echo ${job_name}
log_file=${log_dir}/${job_name}_${mode}.log
python ./dso/NN_remote_2d.py $job_name ${noise} ${ratio} ${mode} ${log_file}

ratio_list="1.0 "



# for ratio in ${ratio_list}
# do
#     noise=0.01
#     mode=1

#     log_dir=./log_remote/noise_${noise}_dataratio_${ratio}
#     mkdir -p ${log_dir}
#     job_name=Allen_Cahn_2D
#     echo ${job_name}
#     log_file=${log_dir}/${job_name}_${mode}.log
#     python ./dso/NN_remote.py  $job_name ${noise} ${ratio} ${mode} ${log_file}

# done