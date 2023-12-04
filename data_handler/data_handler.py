import os

data_folder = "../data/CWE121_Stack_Based_Buffer_Overflow"

# s01부터 s09까지의 폴더를 순회합니다.
for i in range(1, 10):
    folder_name = f"s{i:02d}"
    folder_path = os.path.join(data_folder, folder_name)

    # 폴더 내의 .C 파일을 가져옵니다.
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".c"):
            file_path = os.path.join(folder_path, file_name)
            # 파일을 열어서 C 코드를 읽어옵니다.
            with open(file_path, 'r') as file:
                c_code = file.read()
            # 읽어온 C 코드를 출력하거나 원하는 작업을 수행합니다.
                print(c_code)

