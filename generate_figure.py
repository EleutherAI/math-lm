import matplotlib.pyplot as plt
import numpy as np

# Set to serif style font
plt.rcParams['font.family'] = 'serif'


code_llama_data = [
    {
        'name': 'Code Llama 7B',
        'params': 7,
        'score': 4.4
    },
    {
        'name': 'Code Llama 34B',
        'params': 34,
        'score': 11.9
    },
]

llemma_data = [
    {
        'name': 'Llemma 7B',
        'params': 7,
        'score': 17.2
    },
    {
        'name': 'Llemma 34B',
        'params': 34,
        'score': 24.1
    },
]

minerva_data = [
    {
        'name': 'Minerva 8B',
        'params': 8,
        'score': 14.1
    },
    {
        'name': 'Minerva 62B',
        'params': 62,
        'score': 27.6
    }
]

colors = {
    'Code Llama': 'red',
    'Llemma': 'blue',
    'Minerva': 'orange'
}

markers = {
    'Code Llama': 'o',
    'Llemma': '^',
    'Minerva': 's'
}

data = {
    'Code Llama': code_llama_data,
    'Llemma': llemma_data,
    'Minerva': minerva_data
}

fig, ax = plt.subplots(figsize=(6, 6))

for model_type in ['Code Llama', 'Llemma', 'Minerva']:
    data_for_model = data[model_type]
    x = [d['params'] for d in data_for_model]
    y = [d['score'] for d in data_for_model]
    labels = [d['name'] for d in data_for_model]
    color = colors[model_type]
    marker = markers[model_type]
    ax.plot(x, y, marker, color=color, markersize=10)
    # Add a dotted line between them
    ax.plot(x, y, '--', color=color, alpha=0.5)
    # Add labels
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), fontsize=12, ha='left' if xi < 20 else 'left', 
                    va='bottom' if xi > 20 else 'top', xytext=(-2, -9) if xi < 20 else (2, 7),
                    textcoords='offset points', color=color, fontweight='bold' if model_type == 'Llemma' else 'normal')

ax.set_title("Model Performance on the MATH Dataset")
ax.set_xlabel("# Params")
ax.set_ylabel("Math Pass@1 (Accuracy)")
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Remove ticks
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
# Set the background of the plot to off-white
ax.set_facecolor('#F5F5F5')
# Remove the frame of the chart
for spine in ax.spines.values():
    spine.set_visible(False)

# Add some padding to the plot xlim and ylim
ax.set_xlim(left=0, right=90)
ax.set_ylim(bottom=0, top=35)

# Set y ticks to have a % sign
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x/100) for x in vals])

plt.show()