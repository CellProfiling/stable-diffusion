import glob

with open("hpa_paths.txt", "w") as f:
    for i in range(1,2176):
        i = str(i)
        paths = glob.glob("/scratch/groups/emmalu/HPA_temp/" + i + "/*")
        for path in paths:
            path = "/".join(path.split("/")[5:]) + "\n"
            f.write(path)