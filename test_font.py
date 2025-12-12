#!/usr/bin/env python3
# test_chinese_simple.py - 最简单的中文显示测试

import matplotlib
matplotlib.use('Agg')  # 非GUI环境
import matplotlib.pyplot as plt
import os

# 1. 直接使用字体文件路径
font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'

# 2. 检查字体文件是否存在
if os.path.exists(font_path):
    print(f"✅ 找到字体文件: {font_path}")
    
    # 3. 添加字体到matplotlib
    from matplotlib.font_manager import fontManager
    fontManager.addfont(font_path)
    
    # 4. 获取字体名称并设置为默认
    from matplotlib.font_manager import FontProperties
    chinese_font = FontProperties(fname=font_path)
    font_name = chinese_font.get_name()
    
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False
    
    print(f"✅ 使用字体: {font_name}")
else:
    print("❌ 字体文件不存在，使用默认字体")

# 5. 创建最简单的测试图表
plt.figure(figsize=(8, 5))
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro-')
plt.title('中文标题测试')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')

# 6. 保存图片
output_file = 'chinese_test_simple.png'
plt.savefig(output_file, dpi=100)

# 7. 检查结果
if os.path.exists(output_file):
    size = os.path.getsize(output_file)
    print(f"\n✅ 图片已生成: {output_file}")
    print(f"   文件大小: {size} 字节")
    print(f"   绝对路径: {os.path.abspath(output_file)}")
    print("\n请查看图片中的中文是否正常显示")
else:
    print("❌ 图片生成失败")

