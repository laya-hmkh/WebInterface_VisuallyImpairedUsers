import matplotlib.pyplot as plt
import numpy as np

# Set up figure with two subplots
fig = plt.figure(figsize=(10, 4))

# (a) Bar Chart: % of Images Described
ax1 = fig.add_subplot(121)  # 1 row, 2 columns, 1st plot
systems = ['Our System', 'NVDA']
percentages = [100, 8]  # Our System: 100%, NVDA: 8%
ax1.bar(systems, percentages, color=['#4CAF50', '#F44336'], width=0.5)
ax1.set_ylim(0, 120)
ax1.set_ylabel('Percentage of Images Described (%)')
ax1.set_title('(a) Image Description Coverage', pad=10)
for i, v in enumerate(percentages):
    ax1.text(i, v + 5, f'{v}%', ha='center', fontsize=10)

# (b) Radar Chart: Effectiveness Across Impairments
ax2 = fig.add_subplot(122, polar=True)  # 1 row, 2 columns, 2nd plot (polar for radar)
impairments = ['Far-sightedness', 'Tunnel-vision', 'Sunshine', 'Total Color Blindness', 'Red-green Color Blindness']
scores = [4.8, 4.6, 4.8, 4.7, 4.7]  # Effectiveness scores out of 5
angles = np.linspace(0, 2 * np.pi, len(impairments), endpoint=False).tolist()
scores += scores[:1]  # Close the loop
angles += angles[:1]
ax2.fill(angles, scores, color='#4CAF50', alpha=0.5)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(impairments, fontsize=8)
ax2.set_ylim(0, 5)
ax2.set_title('(b) Effectiveness Across Impairments', pad=20)

# Adjust layout and save
plt.tight_layout()
plt.savefig('figure1.png', dpi=300, bbox_inches='tight')  # PNG for general use
plt.savefig('figure1.eps', format='eps', dpi=300, bbox_inches='tight')  # EPS for LaTeX
plt.show()