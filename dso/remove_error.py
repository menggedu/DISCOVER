"""
 remove Intel error in the logging file 
"""
import sys
log_file = sys.argv[1] #'./log/debug/result_burgers2_pinn_t.log'

if __name__ == "__main__":
    file_name = open(f'{log_file}', 'r')
    file_lines = file_name.readlines()
    file_name.close()
    new_lines = []
    for line in file_lines:
        if line.startswith("Intel"):
            continue
        new_lines.append(line)
    new_file = open(f'{log_file}', 'w')
    new_file.write("".join(new_lines))