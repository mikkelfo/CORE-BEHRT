from matplotlib.ticker import ScalarFormatter
import numpy as np
ALPHA_ADAPTED = 0.55
ALPHA_UNADAPTED = 0.2
#ALPHA_UNADAPTED = 0.3
def normalize_scores(scores, normalization_model=0):
    """Normalize scores by subtracting the first score from all scores."""
    return [(score - scores[normalization_model]) if score is not None else None for score in scores]

def create_scatter_plot(ax, positions, scores, stds, label, color, adapted, normalization_model=0):
    """Creates a scatter plot with error bars."""
    scores = normalize_scores(scores, normalization_model)
    for i, (pos, score, std, a) in enumerate(zip(positions, scores, stds, adapted)):
        if score is not None:
            ax.errorbar(pos, score, yerr=std, markersize=8, fmt='o', capsize=2, capthick=0.8,
                        elinewidth=0.8, mfc=color if a else 'w', mec=color, label=label if i == 0 else None,
                        color=color, alpha=1 if a else .2)

def set_spines_visibility(ax, visibility=False):
    """Sets the visibility of the plot spines."""
    for spine in ['right', 'top', 'bottom', 'left']:
        ax.spines[spine].set_visible(visibility)

def style_plot(ax, y_ticks, fontsize, legend_loc, legend=True):
    """Styles the plot and uses scientific notation for the y-axis."""
    set_spines_visibility(ax)

    # Scale y-ticks (assuming your y-values are already scaled by 1e-2)
    ax.set_yticks(y_ticks)

    # Use scientific notation for y-axis
    ax.tick_params(axis='y', which='major', labelsize=fontsize)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, -2))
    ax.yaxis.get_offset_text().set_size(14)
    if legend:
        ax.legend(loc=legend_loc, frameon=True, fontsize=fontsize, ncol=4)
    ax.grid(False)
    ax.grid(axis='y', linestyle='-', alpha=0.3)
def filter_none(positions, scores):
    """Filters out None values from the scores and positions."""
    return zip(*[(pos, score) for pos, score in zip(positions, scores) if score is not None])
def create_line_plot(ax, positions, scores, color, model_adapted=None, normalization_model=0):
    """Creates a line plot."""
    scores = normalize_scores(scores, normalization_model)
    positions, scores = filter_none(positions, scores)
    #if model_adapted is not None:
     #   positions, scores = zip(*[(pos, score) for pos, score, adapted in zip(positions, scores, model_adapted) if adapted])
    ax.plot(positions, scores, color=color, linestyle='--', linewidth=1, alpha=0.5)

def set_model_adapted_colors(ax, model_adapted):
    """Sets the colors of the x-tick labels based on model adaptation."""
    colors = ['k' if adapted else 'grey' for adapted in model_adapted]
    for xtick, color in zip(ax.get_xticklabels(), colors):
        xtick.set_color(color)
def initialize_plot(models, groups, model_distance, group_gap):
    """Initializes the plot with adjusted positions and group labels."""
    positions = np.arange(len(models))
    adjusted_positions = []
    group_labels = []
    group_label_positions = []
    offset = 0

    for group, indices in groups.items():
        group_size = indices.stop - indices.start
        for i in range(group_size):
            adjusted_positions.append(positions[indices.start + i] + offset)
            offset += model_distance
        group_labels.append(group)
        group_label_positions.append(np.mean(adjusted_positions[-group_size:]))
        offset += group_gap  # Adding space after each group
    # Adjust the positions for plotting
    adjusted_positions = np.array(adjusted_positions) - 1.5 * model_distance
    return adjusted_positions, group_labels, group_label_positions

def plot_data(ax, positions, model_adapted, scores,stds, colors, model_distance, normalization_model = 0, line_plot=True):
    """Plots the data for each category."""
    for i, (category, color) in enumerate(colors.items()):
        category_scores = scores[category]
        category_stds = stds[category] 
        if category != 'average':
            create_bar_plot(ax, positions + i * model_distance, category_scores, category_stds, 
                            category.upper(), color, model_adapted, model_distance, normalization_model)
        else:
            create_scatter_plot(ax, positions+2*model_distance , category_scores, category_stds, category.upper(), color, model_adapted, normalization_model)
            if line_plot:
                create_line_plot(ax, positions+2*model_distance , category_scores, color, model_adapted, normalization_model)

def create_bar_plot(ax, positions, scores, stds, label, color, adapted, model_distance, normalization_model = 0):
    """Creates a scatter plot with error bars."""
    scores = normalize_scores(scores, normalization_model)
    for i, (pos, score, std, a) in enumerate(zip(positions, scores, stds, adapted)):
        if score is not None:
            ax.bar(pos, score, color=color, alpha=ALPHA_ADAPTED if a else ALPHA_UNADAPTED, width=model_distance, label=label if i == 0 else None)
def data_to_axes_fraction(ax, data_x):
    """Convert a data x-coordinate to axes fraction."""
    xlim = ax.get_xlim()
    return (data_x - xlim[0]) / (xlim[1] - xlim[0])

def add_bracket_with_text(ax, x_center, width, y_pos_bracket, text, fontsize=12, annotation_distance=0.05):
    x = data_to_axes_fraction(ax, x_center)
    ax.annotate('', xy=(x, y_pos_bracket), xytext=(x, y_pos_bracket-0.00001),
                fontsize=14, ha='left', va='bottom', xycoords='axes fraction', 
                arrowprops=dict(arrowstyle=f'-[, widthB={width}, lengthB=.7', lw=2.0))
    ax.text(x, y_pos_bracket-annotation_distance, text, transform=ax.transAxes, ha='center', va='top', fontsize=fontsize, fontweight='bold', color='black')

def get_scores(model_groups, original_models, scores, inv_model_map):
    new_scores = {disease: [] for disease in scores.keys()}
    for group, new_models in model_groups.items():
        for model in new_models:
            original_model = inv_model_map[model]
            model_index = original_models.index(original_model)
            for key in scores.keys():
                new_scores[key].append(scores[key][model_index])

    return new_scores
def rename_tasks(scores: dict, task_rename:dict)-> dict:
    new_scores = {}
    for key in scores.keys():
        new_key = task_rename[key]
        new_scores[new_key] = scores[key]
    return new_scores