# Updated: April 26, 2025
# 这个脚本用于更新__pycache__目录下的文件的时间戳

import os
import time

# 遍历所有.pyc文件并更新它们的时间戳
def update_pyc_timestamps():
    # 更新根目录下的__pycache__
    if os.path.exists('__pycache__'):
        for filename in os.listdir('__pycache__'):
            if filename.endswith('.pyc'):
                filepath = os.path.join('__pycache__', filename)
                os.utime(filepath, (time.time(), time.time()))
                print(f"更新了 {filepath}")
    
    # 更新model目录下的__pycache__
    if os.path.exists('model/__pycache__'):
        for filename in os.listdir('model/__pycache__'):
            if filename.endswith('.pyc'):
                filepath = os.path.join('model/__pycache__', filename)
                os.utime(filepath, (time.time(), time.time()))
                print(f"更新了 {filepath}")

update_pyc_timestamps()
print("已更新所有模块的缓存文件") 