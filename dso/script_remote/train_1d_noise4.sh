
# folder_name=noise_0.01_1

# folder=${folder_name}
# echo ${folder}

# mkdir -p ./log_remote/${folder}

# job_name=kdv
# echo ${job_name}
# python test_pde.py ${job_name} ${folder}  > ./log_remote/${folder}/result_${job_name}.log 2>&1

# job_name=divide
# echo ${job_name}
# python test_pde.py ${job_name} ${folder}  > ./log_remote/${folder}/result_${job_name}.log 2>&1

# job_name=burgers
# echo ${job_name}
# python test_pde.py  ${job_name} ${folder} > ./log_remote/${folder}/result_${job_name}.log 2>&1

# job_name=chafee
# echo ${job_name}
# python test_pde.py  ${job_name} ${folder} > ./log_remote/${folder}/result_${job_name}.log 2>&1

# job_name=compound
# echo ${job_name}
# python test_pde.py  ${job_name} ${folder} > ./log_remote/${folder}/result_${job_name}.log 2>&1 

# job_name=ac
# echo ${job_name}
# log_file=${log_dir}/${job_name}_${mode}.log
# python test_pde.py ${job_name} ${folder}  > ./log_remote/${folder}/result_${job_name}.log 2>&1
# # 


ratio_list="0.4"



for ratio in ${ratio_list}
do

    folder_name=noise_0.01_${ratio}
    folder=${folder_name}
    job_name=ac
    echo ${folder} $job_name
    log_file=${log_dir}/${job_name}_${mode}.log
    python test_pde.py ${job_name} ${folder}  
    # > ./log_remote/${folder}/result_${job_name}.log 2>&1

done