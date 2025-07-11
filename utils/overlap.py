import os

dir1 = "/users/mbredber/scratch/data/PSZ2/classified/T100kpcSUB/DE"
dir2 = "/users/mbredber/scratch/data/PSZ2/classified/T100kpcSUB/NDE"

files1 = set(os.listdir(dir1))
files2 = set(os.listdir(dir2))

common_files = sorted(files1.intersection(files2))
print("Common files:")
for fname in common_files:
    print(fname)

# count .npy files in each
npy_count1 = sum(1 for f in files1 if f.endswith('.npy'))
npy_count2 = sum(1 for f in files2 if f.endswith('.npy'))

print(f"Number of .npy files in {dir1}: {npy_count1}")
print(f"Number of .npy files in {dir2}: {npy_count2}")

print("Finished")