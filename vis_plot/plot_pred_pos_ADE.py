'''
绘制三种类型运动, 不同的逆扩散step情况下的, 预测准确性
'''
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

# Set the font family to Arial
plt.rcParams['font.family'] = 'Arial'
# Set the font size to 6
plt.rcParams['font.size'] = 6
# Set the axes linewidth to 0.5
plt.rcParams['axes.linewidth'] = 0.5
# Set the major xtick width to 0.5
plt.rcParams['xtick.major.width'] = 0.5
# Set the major ytick width to 0.5
plt.rcParams['ytick.major.width'] = 0.5
# Set the minor xtick width to 0.5
plt.rcParams['xtick.minor.width'] = 0.5
# Set the minor ytick width to 0.5
plt.rcParams['ytick.minor.width'] = 0.5
# Set the xtick direction to 'in'
plt.rcParams['xtick.direction'] = 'in'
# Set the ytick direction to 'in'
plt.rcParams['ytick.direction'] = 'in'
# Disable the top spine
plt.rcParams['axes.spines.top'] = False
# Disable the right spine
plt.rcParams['axes.spines.right'] = False
# Enable the left spine
plt.rcParams['axes.spines.left'] = True
# Enable the bottom spine
plt.rcParams['axes.spines.bottom'] = True
# Set the axes label size to 6
plt.rcParams['axes.labelsize'] = 6
# Set the legend font size to 6
plt.rcParams['legend.fontsize'] = 6
# Disable the legend frame
plt.rcParams['legend.frameon'] = False
# Set the legend handle length to 1
plt.rcParams['legend.handlelength'] = 1
# Set the legend handle text padding to 0.5
plt.rcParams['legend.handletextpad'] = 0.5
# Set the legend label spacing to 0.5
plt.rcParams['legend.labelspacing'] = 0.5
# Set the legend location to 'upper right'
plt.rcParams['legend.loc'] = 'upper right'
# Set the lines linewidth to 0.5
plt.rcParams['lines.linewidth'] = 0.5
# Set the lines markersize to 2
plt.rcParams['lines.markersize'] = 2
# Set the lines marker to 'o'
plt.rcParams['lines.marker'] = 'o'
# Set the lines marker edge width to 0.5
plt.rcParams['lines.markeredgewidth'] = 0.5
# Set the figure DPI to 450
plt.rcParams['figure.dpi'] = 450
# Set the figure size (convert mm to inches)
plt.rcParams['figure.figsize'] = (160/25.4, 45/25.4)
# Set the savefig DPI to 450
plt.rcParams['savefig.dpi'] = 450
# Set the savefig format to 'pdf'
plt.rcParams['savefig.format'] = 'pdf'
# Set the savefig bbox to 'tight'
plt.rcParams['savefig.bbox'] = 'tight'
# Set the savefig pad inches to 0.05
plt.rcParams['savefig.pad_inches'] = 0.05
# Set the PDF font type to 42
plt.rcParams['pdf.fonttype'] = 42
# Set the PDF compression to 9
plt.rcParams['pdf.compression'] = 9
# Use 14 core fonts in PDF
plt.rcParams['pdf.use14corefonts'] = True
# Do not inherit color in PDF
plt.rcParams['pdf.inheritcolor'] = False


# 扩散步数
diffusion_steps = np.array([100 , 50 , 25 , 20 , 10 , 5 , 2 , 1])

# ADE值
ade_values = {
    'Directed Motion': [0.465, 0.816, 2.293, 2.869, 4.182, 4.944, 5.412, 5.534],
    'Directed-Brownian Switching Motion': [0.711, 0.903, 1.634, 1.922, 2.638, 3.070, 3.355, 3.414],
    'Brownian Motion': [1.640, 1.790, 2.252, 2.432, 2.914, 3.237, 3.429, 3.463]
}

# 创建图形和轴
fig, ax = plt.subplots()  # 设置图形大小和分辨率

# 绘制折线图
for motion_type, values in ade_values.items():
    ax.plot(diffusion_steps, values, label=motion_type)

# 设置标题和标签
ax.set_title('ADE Evaluation for Different Motion Types')
ax.set_xlabel('Diffusion Steps')
ax.set_ylabel('ADE Value')

ax.set_xticks(diffusion_steps)
# 设置图例
ax.legend(title='Motion Type')

# 设置横轴和纵轴的刻度字体大小
ax.tick_params(axis='both', which='major')

# 设置网格
# ax.grid(True, which='both', linestyle='--', linewidth=0.25)

# 保存图形为PDF
plt.savefig('motion_evaluation_line_chart.pdf', bbox_inches='tight', pad_inches=0.1)

# 显示图形
plt.show()