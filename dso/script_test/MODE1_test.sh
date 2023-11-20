# DISCOVER example
# Clean and complete data
# numerical differentiation is utilized

folder_name=MODE1

folder=${folder_name}
echo ${folder}

mkdir -p ./log/${folder}

job_name=Burgers
echo ${job_name}
python test_pde.py  ${job_name} ${folder} 
# > ./log/${folder}/result_${job_name}.log 2>&1
