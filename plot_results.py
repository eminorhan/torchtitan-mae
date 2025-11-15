import matplotlib.pyplot as plt

# Data for the validation lines (renamed from original)
vit_7b_val = [0.4328, 0.4028, 0.4016, 0.3952, 0.3879, 0.4064, 0.4075, 0.4249, 0.4525, 0.4501]
# X-ticks for the first line: 100 to 1000 (inclusive) in steps of 100
x_ticks_7b = list(range(100, 1001, 100))

vit_l_val = [1.6804, 0.7486, 0.5698, 0.5030, 0.4643, 0.4616, 0.4478, 0.4361, 0.4423, 0.4329, 0.4595, 0.4326, 0.4263, 0.4383, 0.4520]
# X-ticks for the second line: 100 to 1400 (inclusive) in steps of 100
x_ticks_l = list(range(100, 1501, 100))

# Data for the new training loss lines
vit_7b_train = [1.3112, 0.1997, 0.1733, 0.1549, 0.1433, 0.1352, 0.1282, 0.1230, 0.1185, 0.1149]
vit_l_train = [2.2026, 0.5894, 0.3247, 0.2589, 0.2322, 0.2088, 0.1978, 0.1896, 0.1829, 0.1736, 0.1692, 0.1651, 0.1624, 0.1568, 0.1547]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the validation lines (solid)
plt.plot(x_ticks_7b, vit_7b_val, marker='o', color='blue', linestyle='-', label='ViT-7B - Validation')
plt.plot(x_ticks_l, vit_l_val, marker='s', color='red', linestyle='-', label='ViT-L - Validation')

# Plot the training lines (dashed)
plt.plot(x_ticks_7b, vit_7b_train, marker='o', color='blue', linestyle='--', label='ViT-7B - Training')
plt.plot(x_ticks_l, vit_l_train, marker='s', color='red', linestyle='--', label='ViT-L - Training')

# Set labels and title
plt.xlabel('Training steps')
plt.ylabel('Loss') # Updated y-axis label
plt.title('Loss vs. Training Steps') # Updated title

# Add legend
plt.legend()

# Add grid for better readability
plt.grid(True)

# Define reasonable x-axis ticks to cover both datasets
all_x_ticks = sorted(list(set(x_ticks_7b + x_ticks_l)))
# Show ticks every 200 steps for clarity, ensuring the last tick is included
step = 200
plot_x_ticks = list(range(min(all_x_ticks), max(all_x_ticks) + step, step))
if max(all_x_ticks) not in plot_x_ticks:
    plot_x_ticks.append(max(all_x_ticks))

plt.xticks(plot_x_ticks)

# Ensure layout is clean
plt.tight_layout()

# Display the plot
plt.savefig(f"train_val_loss.jpeg", bbox_inches='tight')