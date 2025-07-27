#!/usr/bin/env bash
#
# run.sh  -  启动 TShape 训练/测试脚本
# 用法：
#   ./run.sh -d DATASET_NAME     # 单数据集
#   ./run.sh -d "A,B,C"          # 多数据集逗号隔开
# 默认数据集: TODS
#

set -e

# ----------- 1. 默认参数 -----------
DEFAULT_DATASETS=("TODS")          # 默认只跑 TODS
PYTHON_BIN="python3"               # 如有需要可改成 python 或指定虚拟环境 python

# ----------- 2. 解析命令行参数 -----------
while getopts ":d:" opt; do
  case "$opt" in
    d)
      IFS=',' read -ra INPUT_DATASETS <<< "$OPTARG"
      ;;
    \?)
      echo "非法选项: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# 若用户未传 -d，则使用默认
if [[ ${#INPUT_DATASETS[@]} -eq 0 ]]; then
  INPUT_DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# 拼成 python 列表字符串，例如 ["UCR","Yahoo"]
printf -v DATASET_STR ',"%s"' "${INPUT_DATASETS[@]}"
DATASET_STR="[${DATASET_STR:1}]"

echo "即将在数据集 ${DATASET_STR} 上运行 TShape ..."

# ----------- 3. 生成临时 main.py -----------
# 用 sed 把 "datasets = [...]" 这一行替换掉
cp main.py main_tmp.py
sed -i "s/^datasets = \[.*\]/datasets = ${DATASET_STR}/" main_tmp.py

# ----------- 4. 运行 -----------
$PYTHON_BIN main_tmp.py

# ----------- 5. 清理 -----------
rm -f main_tmp.py
echo "运行结束，已清理临时文件."