import glob, os

directory = "rPPG/experiments/yousuf-re-run"
distances = ["1", "1.5", "2"]
exercises = ["stat", "star", "jog"]
extensions = ["edf", "mp4", "csv"]
repeats = ["1", "2", "3"]

files = [file for file in os.listdir(directory)]
edf_files = list(filter(lambda x: x.endswith(".EDF"), files))
mp4_files = list(filter(lambda x: x.endswith(".mp4"), files))
csv_files = list(filter(lambda x: x.endswith(".csv"), files))
edf_files.sort()
mp4_files.sort()
csv_files.sort()
files = [edf_files, mp4_files, csv_files]
print(edf_files)
print(mp4_files)
print(csv_files)
print(len(edf_files))
print(len(mp4_files))
assert len(edf_files) == len(mp4_files)
assert len(csv_files) == len(mp4_files)
assert len(edf_files) == len(csv_files)
for i in range(len(edf_files)):
    print(f"{edf_files[i]} {mp4_files[i]} {csv_files[i]}")
for i in range(len(edf_files)):
    print(f"I: {i}, {i//9}, {(i//3)%3}, {i%3}")
    dist, exer, repeat = distances[i//9], exercises[(i//3)%3], repeats[i%3]
    for ext_index in range(3): 
        file,ext = files[ext_index][i], extensions[ext_index]
        print(f"Renamed {directory}/{file} to {directory}/{dist}_{exer}_{repeat}.{ext}")
        os.rename(f"{directory}/{file}", f"{directory}/{dist}_{exer}_{repeat}.{ext}")