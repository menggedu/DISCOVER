
folder_name=debug

folder=${folder_name}
echo ${folder}

mkdir -p ./log_remote/${folder}

job_name=kdv
echo ${job_name}
python test_pde.py remote ${job_name} ${folder}
# > ./log/${folder}/result_${job_name}.log 2>&1

# job_name=divide
# job_name=ac
# echo ${job_name}
# python test_pde.py ${job_name} ${folder} > ./log/${folder}/result_${job_name}.log 2>&1


# job_name=ch_lap
# echo ${job_name}
# python test_pde.py ${job_name} ${folder} > ./log/${folder}/result_${job_name}.log 2>&1
# job_name=ch_1000
# echo ${job_name}
# python test_pde.py ${job_name} ${folder} > ./log/${folder}/result_${job_name}.log 2>&1
# job_name=burgers
# echo ${job_name}
# python test_pde.py windows ${job_name} ${folder}> ./log/${folder}/result_${job_name}.log 2>&1

# job_name=chafee
# echo ${job_name}
# python test_pde.py windows ${job_name} ${folder}> ./log/${folder}/result_${job_name}.log 2>&1

# job_name=compound
# echo ${job_name}
# python test_pde.py windows ${job_name} ${folder}> ./log/${folder}/result_${job_name}.log 2>&1

