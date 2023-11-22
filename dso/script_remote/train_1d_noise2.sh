
folder_name=noise_0.05_0.2

folder=${folder_name}
echo ${folder}

mkdir -p ./log_remote/${folder}

job_name=kdv
echo ${job_name}
python test_pde.py ${job_name} ${folder}  > ./log_remote/${folder}/result_${job_name}.log 2>&1

job_name=divide
echo ${job_name}
python test_pde.py ${job_name} ${folder}  > ./log_remote/${folder}/result_${job_name}.log 2>&1

job_name=burgers
echo ${job_name}
python test_pde.py  ${job_name} ${folder} > ./log_remote/${folder}/result_${job_name}.log 2>&1

job_name=chafee
echo ${job_name}
python test_pde.py  ${job_name} ${folder} > ./log_remote/${folder}/result_${job_name}.log 2>&1

job_name=compound
echo ${job_name}
python test_pde.py  ${job_name} ${folder} > ./log_remote/${folder}/result_${job_name}.log 2>&1 

