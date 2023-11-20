
# job_name=Kdv
# echo ${job_name}
# python -m dso.task.pde.NN $job_name 0

# job_name=divide
# echo ${job_name}
# python -m dso.task.pde.NN $job_name 0

# job_name=Burgers
# echo ${job_name}
# CUDA_VISIBLE_DEVICES=1 python -m dso.task.pde.NN $job_name 0
job_name=chafee-infante
echo ${job_name}
CUDA_VISIBLE_DEVICES=1 python -m dso.task.pde.NN $job_name 0
job_name=PDE_compound
echo ${job_name}
CUDA_VISIBLE_DEVICES=1 python -m dso.task.pde.NN $job_name 0