
# job_name=Kdv
# echo ${job_name}
# python -m dso.task.pde.NN $job_name 0

job_name=PDE_divide
echo ${job_name}
python -m dso.task.pde.NN $job_name 0

job_name=burgers
echo ${job_name}

job_name=chafee
echo ${job_name}

job_name=compound
echo ${job_name}