from pathlib import Path

i = 0
for file in Path(r"E:\step_files").rglob('*.*'):
    if file.suffix.lower() not in ['.stp', '.step']:
        i+=1
        print(file.name)
print(i)