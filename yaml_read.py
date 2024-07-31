"""
yaml 文件的读写
"""
import os.path

import yaml  # 对于多个参数, 写成一个config 更好


def read_yaml(yaml_file):
    """读取 yaml 文件

    Args:
        yaml_file: str
            yaml path

    Returns:

    """
    with open(yaml_file, encoding="utf-8", mode="r") as f:
        return yaml.load(stream=f.read(), Loader=yaml.FullLoader)


def write_yaml(yaml_file, data):
    """写数据到 yaml 文件

    Args:
        yaml_file: str
            yaml path

        data: 任意类型
            写入的数据

    Returns:
        None

    """

    with open(yaml_file, encoding="utf-8", mode="w") as f:

        # yaml.dump(data, stream=f_pre, allow_unicode=True)
        yaml.dump(data, stream=f)


if __name__ == "__main__":
    # read_yaml("config.yaml.yaml")
    data = {'house': [{'age': 696}, {'height': [1, 2]}]}
    write_yaml(os.path.join(r"C:\Users\25760\Desktop\jicui_doc_0910\time-series-anomaly-detection-main\train_predcit_analysis\module", "config.yaml.yaml"), data)



