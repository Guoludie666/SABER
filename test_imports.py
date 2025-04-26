# Updated: April 26, 2025
# 这个文件用于测试导入模块和更新pyc文件

print("开始测试导入模块...")

# 导入根模块
try:
    import utils
    print("成功导入 utils")
except Exception as e:
    print(f"导入 utils 失败: {e}")

try:
    import arguments
    print("成功导入 arguments")
except Exception as e:
    print(f"导入 arguments 失败: {e}")

try:
    import distance
    print("成功导入 distance")
except Exception as e:
    print(f"导入 distance 失败: {e}")

# 导入model模块
try:
    import model.utils
    print("成功导入 model.utils")
except Exception as e:
    print(f"导入 model.utils 失败: {e}")

try:
    import model.token_classification
    print("成功导入 model.token_classification")
except Exception as e:
    print(f"导入 model.token_classification 失败: {e}")

try:
    import model.question_answering
    print("成功导入 model.question_answering")
except Exception as e:
    print(f"导入 model.question_answering 失败: {e}")

try:
    import model.sequence_classification
    print("成功导入 model.sequence_classification")
except Exception as e:
    print(f"导入 model.sequence_classification 失败: {e}")

try:
    import model.multiple_choice
    print("成功导入 model.multiple_choice")
except Exception as e:
    print(f"导入 model.multiple_choice 失败: {e}")

try:
    import model.deberta
    print("成功导入 model.deberta")
except Exception as e:
    print(f"导入 model.deberta 失败: {e}")

try:
    import model.debertaV2
    print("成功导入 model.debertaV2")
except Exception as e:
    print(f"导入 model.debertaV2 失败: {e}")

print("测试完成") 