import json
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    model_path = "/home/elwa/research/hierarchical/trained_models/varying_size/real/VGG19-5-pretrained-unfrozen/set_6/validation_results/varying_size/correct_per_action.json"
    with open(model_path, 'r') as file:
        data = json.load(file)

    # Process the data
    numerosities = range(9)
    actions = ['-', '<', '>']
    performance = {action: [] for action in actions}

    for numerosity in numerosities:
        for action in actions:
            key = f"{numerosity}{action}"
            if key in data:
                performance[action].append(data[key][2])
            else:
                performance[action].append(0)  # Fill with 0 if data is missing

    # Plot the data
    bar_width = 0.2
    x = np.arange(len(numerosities))

    plt.figure(figsize=(12, 8))

    for i, action in enumerate(actions):
        plt.bar(x + i * bar_width, performance[action], width=bar_width, label=action)

    plt.xlabel('Numerosity')
    plt.ylabel('Performance (%)')
    plt.title('Performance by Numerosity and Action')
    plt.xticks(x + bar_width, numerosities)
    plt.legend()
    plt.tight_layout()
    plt.savefig("perf_elementwise.png")
