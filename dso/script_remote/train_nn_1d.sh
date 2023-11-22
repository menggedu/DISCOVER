ratio_list=(0.2 0.6 0.8 1.0)

noise=0.1
mode=0

log_dir=./log_remote/noise_${noise}_dataratio_${ratio}
mkdir -p ${log_dir}

for ratio in $ratio_list
do
    job_name=Allen_Cahn_2D
    echo ${job_name}
    log_file=${log_dir}/${job_name}_${mode}.log
    python ./dso/NN_remote.py  $job_name ${noise} ${ratio} ${mode} ${log_file}

done