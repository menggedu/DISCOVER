

# ratio_list="1.0"
# noise_list="0.01"
# datasets="divide chafee compound kdv burgers"
# datasets="compound divide"
# datasets="divide"
# for dataset in ${datasets}
# do
#   for ratio in ${ratio_list}
#   do
#     for noise in ${noise_list}
#     do
#         folder_name=noise_${noise}_${ratio}
#         folder=${folder_name}
#         # echo ${folder}

#         mkdir -p ./log_local/${folder}

#         job_name=${dataset}
#         echo ${job_name}_${folder}
#         python test_pde.py ${job_name} ${folder}  > ./log_local/${folder}/result_${job_name}.log 2>&1
#     done
#   done
# done

# PI_noise

job_name=Burgers2
noise_levels="0.5"
for noise in ${noise_levels}
do
  folder=noise_${noise}
  echo ${job_name}_${folder}
  mkdir -p ./log_pinn/${folder}
  python test_pde.py ${job_name} ${folder}  > ./log_pinn/${folder}/result_${job_name}.log 2>&1

done
