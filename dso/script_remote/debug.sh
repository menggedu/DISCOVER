
folder_name=RRE

folder=${folder_name}
echo ${folder}

mkdir -p ./log/${folder}

job_name=compound_noise
job_name=kdv
# job_name=chafee_noise
echo ${job_name}
# python test_pde.py  ${job_name} ${folder}
# > ./log/${folder}/result_${job_name}.log 2>&1

# job_name=divide
# job_name=ac
# echo ${job_name}
# python test_pde.py ${job_name} ${folder} > ./log/${folder}/result_${job_name}.log 2>&1


job_name=ch_lap
echo ${job_name}
python test_pde.py ${job_name} ${folder} 
# > ./log/${folder}/result_${job_name}.log 2>&1
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

# job_name=Cahn_Hilliard_2D
# echo $job_name
# python ./dso/NN_remote.py $job_name 0.1 1 0 ./log_remote/debug
