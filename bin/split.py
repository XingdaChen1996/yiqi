import os
import pandas as pd
from sklearn.model_selection import train_test_split


def add_suffix_to_filename(file_path, suffix):
    # 分离文件路径的根和扩展名
    root, ext = os.path.splitext(file_path)
    # 创建新的文件名
    new_file_path = f"{root}{suffix}{ext}"
    return new_file_path


if __name__ == '__main__':
    path = r'..\舱室声压级20240729.xlsx'
    test_size = 0.1
    random_state = 0
    X = pd.read_excel(path, sheet_name='input')
    y = pd.read_excel(path, sheet_name='output')

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        shuffle=True)

    path_train = add_suffix_to_filename(path, "_train")
    path_test = add_suffix_to_filename(path, "_test")

    with pd.ExcelWriter(path_train) as writer:
        X_train.to_excel(writer, sheet_name='input', index=True)
        y_train.to_excel(writer, sheet_name='output', index=True)

    with pd.ExcelWriter(path_test) as writer:
        X_test.to_excel(writer, sheet_name='input', index=True)
        y_test.to_excel(writer, sheet_name='output', index=True)

    print('end')
