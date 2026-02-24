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
pretrained_128_3d_tr = [4.1187, 2.4609, 2.3419]
pretrained_128_3d_val = [0.0338, 0.0380]

pretrained_256_3d_tr = []
pretrained_256_3d_val = []

pretrained_480_3d_tr = [5.9727, 3.1452, 2.6860, 2.3893, 2.1718, 2.0590, 1.9760, 1.9339, 1.8241, 1.7848, 1.7474, 1.6635, 1.6425, 1.5816, 1.5164, 1.5156, 1.5054, 1.4359, 1.4328, 1.3786, 1.3948]
pretrained_480_3d_val = [0.0367, 0.0458, 0.0506, 0.0790, 0.0679, 0.0978, 0.0753, 0.0852, 0.1063, 0.0920, 0.1226, 0.1062, 0.1230, 0.1266, 0.1441, 0.1427, 0.1424, 0.1452, 0.1509, 0.1532]

# 3D no bf16
pretrained_128_3d_no_bf16_tr = []
pretrained_128_3d_no_bf16_val = []

pretrained_256_3d_no_bf16_tr = []
pretrained_256_3d_no_bf16_val = []

pretrained_480_3d_no_bf16_tr = [4.0902, 2.3928, 2.0257, 1.8098, 1.6501, 1.4478, 1.3114, 1.1980, 1.1214, 1.0227, 1.0003, 0.9535, 0.8973, 0.8819, 0.8375, 0.8135, 0.8006, 0.7799, 0.7653, 0.7535, 0.7502]
pretrained_480_3d_no_bf16_val = [0.0531, 0.0466, 0.0443, 0.0604, 0.0990, 0.0831, 0.1061, 0.1158, 0.1307, 0.1302, 0.1352, 0.1542, 0.1537, 0.1676, 0.1631, 0.1793, 0.1743, 0.1805, 0.1840, 0.1854]




# Organize into a list of tuples to iterate
models = [
    ('pretrained_2d', pretrained_2d_tr, pretrained_2d_val, pretrained_2d_x, '#56B4E9'),
    ('scratch_2d', scratch_2d_tr, scratch_2d_val, scratch_2d_x, '#E69F00'),
    ('scratch_2d_longer', scratch_2d_longer_tr, scratch_2d_longer_val, scratch_2d_longer_x, '#009E73'),
    ('pretrained_3d', pretrained_3d_tr, pretrained_3d_val, pretrained_3d_x, '#F0E442'),
    ('pretrained_3d_no_bf16', pretrained_3d_no_bf16_tr, pretrained_3d_no_bf16_val, pretrained_3d_no_bf16_x, '#0072B2'),
    ('scratch_3d_no_bf16', scratch_3d_no_bf16_tr, scratch_3d_no_bf16_val, scratch_3d_no_bf16_x, '#D55E00')
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for idx, (name, tr, val, x, c) in enumerate(models):
    # Plot training metrics with all x values
    ax1.plot(x, tr, label=name, color=c, linewidth=2)
    # Plot validation metrics excluding the first iteration (x[1:])
    ax2.plot(x[1:], val, label=name, color=c, linewidth=2)

# Training Subplot
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Training Loss')
ax1.set_ylim([0, 4])
ax1.set_xlim([0, 10000])
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(frameon=False)

# Validation Subplot
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Validation mIoU')
ax2.set_ylim([0, 0.4])
ax2.set_xlim([0, 10000])
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(frameon=False)

plt.tight_layout()
plt.savefig('metrics_input_size.png', dpi=300)