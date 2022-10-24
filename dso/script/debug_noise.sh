# job_name=PDE_divide
# # echo ${job_name}
# normalize=min_max
# log_file=${log_dir}/${job_name}_${normalize}.log
# python train_NN.py --data-name $job_name --noise-level 0.01 --train-ratio 0.4 --normalize-type ${normalize} --maxit 100

# job_name=Cahn_Hilliard_2D
# echo ${job_name}

# log_file=${log_dir}/${job_name}_${normalize}.log
# python train_NN.py --data-name $job_name --noise-level 0.01 --hidden-num 248 --num-layer 1 --train-ratio 0.4 --display-step 1 --maxit 100

folder_name=debug

folder=${folder_name}
echo ${folder}

mkdir -p ./log_local/${folder}

job_name=compound_noise
echo ${job_name}
python test_pde.py ${job_name} ${folder}  