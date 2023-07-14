ratio_list="0.1"
# 1.0 0.8 0.6 
for ratio in $ratio_list
do
echo $ratio
noise=0.01
mode=1
log_dir=./log_local/noise_${noise}_dataratio_${ratio}
mkdir -p ${log_dir}

job_name=Kdv
echo ${job_name}
log_file=${log_dir}/${job_name}_${mode}.log
CUDA_VISIBLE_DEVICES=1 python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

job_name=PDE_divide
echo ${job_name} 
log_file=${log_dir}/${job_name}_${mode}.log
CUDA_VISIBLE_DEVICES=1 python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

job_name=Burgers
echo ${job_name}
log_file=${log_dir}/${job_name}_${mode}.log
CUDA_VISIBLE_DEVICES=1 python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

job_name=chafee-infante
echo ${job_name}
log_file=${log_dir}/${job_name}_${mode}.log
CUDA_VISIBLE_DEVICES=1 python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}

job_name=PDE_compound
echo ${job_name}
log_file=${log_dir}/${job_name}_${mode}.log
CUDA_VISIBLE_DEVICES=1 python -m dso.NN_local $job_name ${noise} ${ratio} ${mode} ${log_file}
done