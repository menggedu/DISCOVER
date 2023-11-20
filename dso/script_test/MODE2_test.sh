# R_discover example 
# discovering process + embedding process
# deal with sparse and noisy data with automatic differentiation

job_name=Burgers2

folder=MODE2
# ${job_name}_pinn
mkdir -p ./log_pinn/${folder}

noise_levels="0.50"

pde=${job_name}_noise_${noise_levels}
echo ${pde}
python test_pde.py ${pde}  ${folder} 
# > ./log_pinn/${folder}/result_${pde}.log 2>&1
