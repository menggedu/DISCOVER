# DISCOVER example
# Numerical differentiation is utilized to evaluate partial derivatives.
# Data type:
# (1) Clean and complete data
# (2) Preprocessd data on regularized grids (Smoothing with NNs). 


# Build folder for saving searching results
folder_name=MODE1
folder=${folder_name}
mkdir -p ./log/${folder}

# KdV equation discovery
job_name=KdV
echo ${job_name}
python test_pde.py  ${job_name} ${folder}

# Chafee-Infante equation discovery
job_name=Chafee
echo ${job_name}
python test_pde.py  ${job_name} ${folder}


# PDE_Compound equation discovery
job_name=Compound
echo ${job_name}
# python test_pde.py  ${job_name} ${folder}

# Burgers equation discovery
job_name=Burgers
echo ${job_name}
# python test_pde.py  ${job_name} ${folder} 

# PDE_divide equation discovery
job_name=Divide
echo ${job_name}
# python test_pde.py ${job_name} ${folder} 


