import matplotlib.pyplot as plt


x = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

# 2D 
pretrained_128_2d_tr = [4.5225, 1.2757, 0.5282, 0.4987, 0.4722, 0.4729, 0.4622, 0.4522, 0.4438, 0.4336, 0.4342, 0.4259, 0.4223, 0.4160, 0.4143, 0.4146, 0.4069, 0.4063, 0.4069, 0.4019, 0.4014]
pretrained_128_2d_val = [0.1972, 0.2251, 0.2003, 0.2151, 0.2159, 0.2245, 0.2040, 0.2226, 0.2324, 0.2158, 0.2298, 0.2395, 0.2337, 0.2261, 0.2428, 0.2350, 0.2428, 0.2406, 0.2414, 0.2418]

pretrained_256_2d_tr = [4.4814, 1.2056, 0.4082, 0.3591, 0.3491, 0.3318, 0.3193, 0.3133, 0.3067, 0.2996, 0.2982, 0.2910, 0.2864, 0.2857, 0.2812, 0.2816, 0.2781, 0.2770, 0.2755, 0.2714, 0.2707]
pretrained_256_2d_val = [0.2105, 0.2724, 0.3012, 0.2993, 0.2835, 0.3044, 0.3060, 0.3074, 0.3047, 0.3162, 0.3282, 0.3233, 0.3262, 0.3215, 0.3303, 0.3260, 0.3301, 0.3276, 0.3335, 0.3338]

pretrained_480_2d_tr = [4.3848, 1.1910, 0.3436, 0.2931, 0.2752, 0.2621, 0.2530, 0.2505, 0.2391, 0.2312, 0.2290, 0.2241, 0.2196, 0.2160, 0.2139, 0.2102, 0.2066, 0.2058, 0.2041, 0.2033, 0.2003]
pretrained_480_2d_val = [0.2703, 0.3187, 0.3284, 0.3125, 0.2965, 0.3320, 0.3165, 0.2973, 0.3473, 0.3435, 0.3279, 0.3410, 0.3552, 0.3473, 0.3547, 0.3558, 0.3580, 0.3530, 0.3675, 0.3671]

# 3D 
pretrained_128_3d_tr = [4.1187, 2.4609, 2.3419, 2.3928, 2.3723, 2.2849, 2.3472, 2.3573, 2.4470, 2.3498, 2.3922, 2.2985, 2.3153, 2.3324, 2.2450, 2.3955, 2.2080, 2.1605, 2.1592, 2.0489, 2.0541]
pretrained_128_3d_val = [0.0338, 0.0380, 0.0363, 0.0195, 0.0362, 0.0300, 0.0333, 0.0271, 0.0363, 0.0368, 0.0350, 0.0332, 0.0358, 0.0267, 0.0334, 0.0363, 0.0318, 0.0462, 0.0368, 0.0472]

pretrained_256_3d_tr = [4.1040, 2.3271, 2.0613, 2.0224, 2.1255, 2.1409, 2.0967, 2.1720, 2.3822, 2.4728, 2.1737, 2.1853, 2.6261, 1.9955, 1.9848, 1.8848, 2.0209, 2.1271, 1.7511, 1.8604, 1.6376]
pretrained_256_3d_val = [0.0328, 0.0264, 0.0521, 0.0483, 0.0459, 0.0466, 0.0485, 0.0552, 0.0461, 0.0487, 0.0470, 0.0641, 0.0522, 0.0468, 0.0660, 0.0787, 0.0777, 0.0777, 0.0806, 0.0803]

pretrained_480_3d_tr = [5.9727, 3.1452, 2.6860, 2.3893, 2.1718, 2.0590, 1.9760, 1.9339, 1.8241, 1.7848, 1.7474, 1.6635, 1.6425, 1.5816, 1.5164, 1.5156, 1.5054, 1.4359, 1.4328, 1.3786, 1.3948]
pretrained_480_3d_val = [0.0367, 0.0458, 0.0506, 0.0790, 0.0679, 0.0978, 0.0753, 0.0852, 0.1063, 0.0920, 0.1226, 0.1062, 0.1230, 0.1266, 0.1441, 0.1427, 0.1424, 0.1452, 0.1509, 0.1532]

# 3D no bf16
pretrained_128_3d_no_bf16_tr = [4.3334, 2.1071, 1.7782, 1.6233, 1.4777, 1.3563, 1.2760, 1.1760, 1.1105, 1.0169, 0.9524, 0.8684, 0.8152, 0.7847, 0.7667, 0.7272, 0.6925, 0.6690, 0.6604, 0.6355, 0.6314]
pretrained_128_3d_no_bf16_val = [0.0129, 0.0405, 0.0406, 0.0236, 0.0295, 0.0364, 0.0315, 0.0404, 0.0547, 0.0668, 0.0660, 0.0694, 0.0725, 0.0756, 0.0661, 0.0716, 0.0617, 0.0664, 0.0625, 0.0626]

pretrained_256_3d_no_bf16_tr = [4.1113, 2.0695, 1.6323, 1.3860, 1.2516, 1.0418, 0.8597, 0.7957, 0.7484, 0.7066, 0.6328, 0.5867, 0.5618, 0.5392, 0.5078, 0.4853, 0.4572, 0.4299, 0.4294, 0.4040, 0.4041]
pretrained_256_3d_no_bf16_val = [0.0413, 0.0529, 0.0470, 0.0509, 0.0800, 0.0869, 0.0938, 0.0966, 0.1069, 0.1120, 0.1240, 0.1214, 0.1268, 0.1328, 0.1305, 0.1323, 0.1383, 0.1416, 0.1472, 0.1476]

pretrained_480_3d_no_bf16_tr = [4.0902, 2.3928, 2.0257, 1.8098, 1.6501, 1.4478, 1.3114, 1.1980, 1.1214, 1.0227, 1.0003, 0.9535, 0.8973, 0.8819, 0.8375, 0.8135, 0.8006, 0.7799, 0.7653, 0.7535, 0.7502]
pretrained_480_3d_no_bf16_val = [0.0531, 0.0466, 0.0443, 0.0604, 0.0990, 0.0831, 0.1061, 0.1158, 0.1307, 0.1302, 0.1352, 0.1542, 0.1537, 0.1676, 0.1631, 0.1793, 0.1743, 0.1805, 0.1840, 0.1854]

# Drop first iteration for validation metrics
x_val = x[1:]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colors: 2D (blues), 3D (reds), 3D no bf16 (greens)
color_2d = ['#99ccff', '#3366cc', '#000066']
color_3d = ['#ff9999', '#cc0000', '#660000']
color_3d_no_bf16 = ['#99ff99', '#00cc00', '#003300']

# Training Loss Plot
ax1 = axes[0]
ax1.plot(x, pretrained_128_2d_tr, color=color_2d[0], label='2d_128', linewidth=2)
ax1.plot(x, pretrained_256_2d_tr, color=color_2d[1], label='2d_256', linewidth=2)
ax1.plot(x, pretrained_480_2d_tr, color=color_2d[2], label='2d_480', linewidth=2)

ax1.plot(x, pretrained_128_3d_tr, color=color_3d[0], label='3d_128', linewidth=2)
ax1.plot(x, pretrained_256_3d_tr, color=color_3d[1], label='3d_256', linewidth=2)
ax1.plot(x, pretrained_480_3d_tr, color=color_3d[2], label='3d_480', linewidth=2)

ax1.plot(x, pretrained_128_3d_no_bf16_tr, color=color_3d_no_bf16[0], label='3d_128_no_bf16', linewidth=2)
ax1.plot(x, pretrained_256_3d_no_bf16_tr, color=color_3d_no_bf16[1], label='3d_256_no_bf16', linewidth=2)
ax1.plot(x, pretrained_480_3d_no_bf16_tr, color=color_3d_no_bf16[2], label='3d_480_no_bf16', linewidth=2)

ax1.set_ylim([0, 4])
ax1.set_xlim([0, 2000])

ax1.set_xlabel('Iterations')
ax1.set_ylabel('Training loss')
ax1.legend(frameon=False)
ax1.grid(True)

# Validation mIoU Plot
ax2 = axes[1]
ax2.plot(x_val, pretrained_128_2d_val, color=color_2d[0], label='2d_128', linewidth=2)
ax2.plot(x_val, pretrained_256_2d_val, color=color_2d[1], label='2d_256', linewidth=2)
ax2.plot(x_val, pretrained_480_2d_val, color=color_2d[2], label='2d_480', linewidth=2)

ax2.plot(x_val, pretrained_128_3d_val, color=color_3d[0], label='3d_128', linewidth=2)
ax2.plot(x_val, pretrained_256_3d_val, color=color_3d[1], label='3d_256', linewidth=2)
ax2.plot(x_val, pretrained_480_3d_val, color=color_3d[2], label='3d_480', linewidth=2)

ax2.plot(x_val, pretrained_128_3d_no_bf16_val, color=color_3d_no_bf16[0], label='3d_128_no_bf16', linewidth=2)
ax2.plot(x_val, pretrained_256_3d_no_bf16_val, color=color_3d_no_bf16[1], label='3d_256_no_bf16', linewidth=2)
ax2.plot(x_val, pretrained_480_3d_no_bf16_val, color=color_3d_no_bf16[2], label='3d_480_no_bf16', linewidth=2)

ax2.set_ylim([0, 0.4])
ax2.set_xlim([0, 2000])

ax2.set_xlabel('Iterations')
ax2.set_ylabel('Validation mIoU')
#ax2.legend(frameon=False)
ax2.grid(True)

plt.tight_layout()
plt.savefig('metrics_inputsize.png')