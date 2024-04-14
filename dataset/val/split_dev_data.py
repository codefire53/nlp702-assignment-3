filename = './UBC_subtask3_dev_1.txt'
dialects = ['arz_Arab', 'ars_Arab', 'ajp_Arab']
split_sizes = [100, 100, 200]
with open(filename, 'r') as f:
    lines = f.readlines()
curr_idx = 0
for idx in range(len(split_sizes)):
    dialect = dialects[idx]
    split_size = split_sizes[idx]
    subset = lines[curr_idx:curr_idx+split_size]
    prefix, suffix = filename.rsplit('.', 1)
    subset_filename = f"{prefix}#{dialect}.{suffix}"
    with open(subset_filename, 'w') as f:
        for row in subset:
            f.write(row.strip()+'\n')
    curr_idx += split_size