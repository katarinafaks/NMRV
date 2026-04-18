
import numpy as np
from matplotlib import pyplot as plt

def plot3():
    
    iterations = []
    failures = []
    avg_overlaps = []
    with open('tests_/bolt_results.txt', 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            iter_line = lines[i]
            avg_line = lines[i+1]
            # Extract number of iterations and failures
            iter_num = int(iter_line.split(',')[0].split(':')[1].strip())
            fail_num = int(iter_line.split(',')[1].split('Tracker failed')[1].split('times')[0].strip())
            avg_overlap = float(avg_line.split(':')[1].strip())
            iterations.append(iter_num)
            failures.append(fail_num)
            avg_overlaps.append(avg_overlap)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    # Bar chart for failures
    bars = ax1.bar(iterations, failures, color='lightblue', label='Number of failures')

    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Number of failures', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, max(failures) + 2)
    ax1.set_xticks(iterations)
    # Add value labels above bars
    for bar, v in zip(bars, failures):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.1, str(v), color='blue', ha='center')

    # Line plot for average overlap
    ax2 = ax1.twinx()
    ax2.plot(iterations, avg_overlaps, color='orange', marker='o', label='Average overlap')
    ax2.set_ylabel('Average overlap', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)

    plt.title('Iterations vs failures and average overlap')
    fig.legend(loc='upper right', bbox_to_anchor=(0.5,1), bbox_transform=ax1.transAxes)
    plt.show()
# plot3()

def plot_gymnastics_results():
    failures = []
    avg_overlaps = []
    alpha_labels = []
    with open('tests_/bolt_results.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines), 4):
            alpha_line = lines[i]
            overlap_line = lines[i+3]
            # Parse alpha value
            alpha_value = alpha_line.split('Alpha:')[1].split(',')[0].strip()
            alpha_labels.append(round(float(alpha_value), 2))
            # Parse average overlap
            avg_overlap = float(overlap_line.split(':')[-1].strip())
            avg_overlaps.append(avg_overlap)

    x = np.arange(len(alpha_labels))
    fig, ax1 = plt.subplots(figsize=(max(8, len(alpha_labels)*0.6), 5))
    ax1.set_xlabel('Alpha value')
    ax1.set_ylabel('')
    ax1.set_xticks(x)
    ax1.set_xticklabels(alpha_labels, rotation=45)
    ax1.set_yticks([])  # Hide y-ticks for the bar axis

    # Emphasize the average overlap line
    ax2 = ax1.twinx()
    ax2.plot(x, avg_overlaps, color='orange', marker='o', linewidth=3, markersize=8, label='Average overlap')
    ax2.set_ylabel('Average overlap', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)

    plt.title('Average overlap per alpha', fontsize=14)
    fig.legend(loc='upper right', bbox_to_anchor=(0.5,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.show()

def plot_avg_overlap():
    avg_overlaps = []
    alpha_labels = []
    with open('tests_/car_alpha_results.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines), 4):
            alpha_line = lines[i]
            overlap_line = lines[i+3]
            # Parse alpha value
            alpha_value = alpha_line.split('Alpha:')[1].split(',')[0].strip()
            alpha_labels.append(round(float(alpha_value), 2))
            # Parse average overlap
            avg_overlap = float(overlap_line.split(':')[-1].strip())
            avg_overlaps.append(avg_overlap)

    x = np.arange(len(alpha_labels))
    fig, ax = plt.subplots(figsize=(max(6, len(alpha_labels)*0.6), 5))
    ax.plot(x, avg_overlaps, color='orange', marker='o', linewidth=3, markersize=8, label='Average overlap')
    ax.set_xlabel('Alpha value')
    ax.set_ylabel('Average overlap', color='red', fontsize=12)
    ax.tick_params(axis='y', labelcolor='red')
    # ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(alpha_labels, rotation=45)
    plt.title('Average overlap per alpha', fontsize=14)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
# plot_avg_overlap()

def plot_avg_overlap_sigma():
    avg_overlaps = []
    sigma_labels = []
    with open('tests_/gymnastics_sigma_results.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines), 4):
            alpha_line = lines[i]
            overlap_line = lines[i+3]
            # Parse sigma value
            sigma_value = alpha_line.split('Sigma:')[1].split(',')[0].strip()
            sigma_labels.append(round(float(sigma_value), 2))
            # Parse average overlap
            avg_overlap = float(overlap_line.split(':')[-1].strip())
            avg_overlaps.append(avg_overlap)

    x = np.arange(len(sigma_labels))
    fig, ax = plt.subplots(figsize=(max(6, len(sigma_labels)*0.6), 5))
    ax.plot(x, avg_overlaps, color='orange', marker='o', linewidth=3, markersize=8, label='Average overlap')
    ax.set_xlabel('Sigma value')
    ax.set_ylabel('Average overlap', color='red', fontsize=12)
    ax.tick_params(axis='y', labelcolor='red')
    # ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(sigma_labels, rotation=45)
    plt.title('Average overlap per sigma', fontsize=14)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
# plot_avg_overlap_sigma()

def plot():
    avg_overlaps = []
    failures = []
    alpha_labels = []
    with open('tests_/gymnastics_alpha_results.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines), 4):
            alpha_line = lines[i]
            fail_line = lines[i+2]
            overlap_line = lines[i+3]
            # Parse alpha value
            alpha_value = alpha_line.split('Alpha:')[1].split(',')[0].strip()
            alpha_labels.append(round(float(alpha_value), 2))
            # Parse failures
            fail_num = int(fail_line.split('Tracker failed')[1].split('times')[0].strip())
            failures.append(fail_num)
            # Parse average overlap
            avg_overlap = float(overlap_line.split(':')[-1].strip())
            avg_overlaps.append(avg_overlap)

    x = np.arange(len(alpha_labels))
    fig, ax1 = plt.subplots(figsize=(max(6, len(alpha_labels)*0.6), 5))
    bars = ax1.bar(x, failures, color='lightblue', label='Number of failures')
    ax1.set_xlabel('Alpha value')
    ax1.set_ylabel('Number of failures', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(alpha_labels, rotation=45)
    for bar, v in zip(bars, failures):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.1, str(v), color='blue', ha='center')

    ax2 = ax1.twinx()
    ax2.plot(x, avg_overlaps, color='orange', marker='o', linewidth=3, markersize=8, label='Average overlap')
    ax2.set_ylabel('Average overlap', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    # ax2.set_ylim(0, 1)

    plt.title('Failures and average overlap per alpha', fontsize=14)
    fig.legend(loc='upper right', bbox_to_anchor=(0.5,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.show()
plot()