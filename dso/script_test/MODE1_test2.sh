# DISCOVER example
# Numerical differentiation is utilized to evaluate partial derivatives.
# Data type:
# (1) Clean and complete data
# (2) Preprocessd data on regularized grids (Smoothing with NNs). 


# Build folder for saving searching results
folder_name=MODE1
folder=${folder_name}
mkdir -p ./log/${folder}


# job_name=wave
# echo ${job_name}
# python test_pde.py  ${job_name} ${folder}

job_name=wave_dyn
echo ${job_name}
python test_pde.py  ${job_name} ${folder}




