folder_name=ablation

folder=${folder_name}
echo ${folder}

mkdir -p ./log/${folder}


job_name=burgers2_sample_0.5
echo ${job_name}
python test_pde.py ${job_name} ${folder}  > ./log/${folder}/result_${job_name}.log 2>&1


job_name=burgers2_sample_0.75
echo ${job_name}
python test_pde.py ${job_name} ${folder}  > ./log/${folder}/result_${job_name}.log 2>&1

job_name=burgers2_0.75
# echo ${job_name}
# python test_pde.py ${job_name} ${folder}  > ./log/${folder}/result_${job_name}.log 2>&1

job_name=burgers2_sample_0.5_500
echo ${job_name}
python test_pde.py ${job_name} ${folder}  > ./log/${folder}/result_${job_name}.log 2>&1