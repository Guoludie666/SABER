# Updated: April 26, 2025
# 这个文件用于测试导入模块和更新pyc文件

print("开始测试导入模块...")

# 导入模块
try:
    from . import utils
    print("成功导入 utils")
except Exception as e:
    print(f"导入 utils 失败: {e}")

try:
    from . import token_classification
    print("成功导入 token_classification")
except Exception as e:
    print(f"导入 token_classification 失败: {e}")

try:
    from . import question_answering
    print("成功导入 question_answering")
except Exception as e:
    print(f"导入 question_answering 失败: {e}")

try:
    from . import sequence_classification
    print("成功导入 sequence_classification")
except Exception as e:
    print(f"导入 sequence_classification 失败: {e}")

try:
    from . import multiple_choice
    print("成功导入 multiple_choice")
except Exception as e:
    print(f"导入 multiple_choice 失败: {e}")

print("测试完成") 