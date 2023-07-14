ratio=0.4
noise=0.01

log_dir=./log_NN/noise_${noise}_dataratio_${ratio}
mkdir -p ${log_dir}

job_name=PDE_divide
job_name=Cahn_Hilliard_2D
echo ${job_name}
normalize=standard
log_file=${log_dir}/${job_name}_${normalize}.log
python train_NN.py --data-name $job_name --noise-level ${noise} --train-ratio ${ratio} \
--normalize-type ${normalize} --hidden-num 128  --num-layer 3 --data-efficient 1 --mode test \
--early-stopper 2000 \
# > ${log_file} 2>&1

# noise=0.005
# log_dir=./log_NN/noise_${noise}_dataratio_${ratio}
# log_file=${log_dir}/${job_name}_${normalize}.log
# python train_NN.py --data-name $job_name --noise-level ${noise} --train-ratio ${ratio} \
# --normalize-type ${normalize} --hidden-num 64 --num-layer 4 --maxit 400000 \
#  --early-stopper 10000 > ${log_file} 2>&1
