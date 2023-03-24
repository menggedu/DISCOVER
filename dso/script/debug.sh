
folder_name=debug

folder=${folder_name}
echo ${folder}

mkdir -p ./log/${folder}

job_name=compound_noise
job_name=kdv
# job_name=chafee_noise
# echo ${job_name}
# python test_pde.py  ${job_name} ${folder}
# > ./log/${folder}/result_${job_name}.log 2>&1

# job_name=divide_noisy
# job_name=ac
# echo ${job_name}
# python test_pde.py ${job_name} ${folder} 
# > ./log/${folder}/result_${job_name}.log 2>&1

# job_name=chafee_t
job_name=kdv_dsr
job_name=burgers2_pinn2
job_name=ks_0.1

# echo ${job_name}
# python test_pde.py ${job_name} ${folder}  > ./log/${folder}/result_${job_name}.log 2>&1

# job_name=burgers2_pinn_t
# job_name=RRE
echo ${job_name}
python test_pde.py ${job_name} ${folder} 
# > ./log/${folder}/result_${job_name}.log 2>&1


# job_name=burgers_dsr
# echo ${job_name}
# python test_pde.py ${job_name} ${folder}> ./log/${folder}/result_${job_name}.log 2>&1

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

