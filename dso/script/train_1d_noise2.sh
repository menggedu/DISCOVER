ratio_list="0.2 0.4 0.6 0.8"
noise_list="0.1"
datasets=" chafee kdv burgers"
# seeds="0 1 2"
# datasets="divide"
for dataset in ${datasets}
do
  for ratio in ${ratio_list}
  do
    for noise in ${noise_list}
    do
        # for seed in ${seeds}
        # do
            folder_name=noise_${noise}_${ratio}
            folder=${folder_name}
            # echo ${folder}

            mkdir -p ./log_local/${folder}

            job_name=${dataset}
            echo ${job_name}_${folder}
            python test_pde.py ${job_name} ${folder}  > ./log_local/${folder}/result_${job_name}.log 2>&1
        # done
    done
  done
done
# job_name=divide
# echo ${job_name}
# python test_pde.py ${job_name} ${folder}  > ./log_remote/${folder}/result_${job_name}.log 2>&1 &

# job_name=burgers
# echo ${job_name}
# python test_pde.py  ${job_name} ${folder} > ./log_remote/${folder}/result_${job_name}.log 2>&1 &

# job_name=chafee
# echo ${job_name}
# python test_pde.py  ${job_name} ${folder} > ./log_remote/${folder}/result_${job_name}.log 2>&1 &

# job_name=compound
# echo ${job_name}
# python test_pde.py  ${job_name} ${folder} > ./log_remote/${folder}/result_${job_name}.log 2>&1 


