from utils.file_reader import read_txt

with open("sample.txt", "rb") as f:
    text = read_txt(f)

print("FILE CONTENT:")
print(text)
