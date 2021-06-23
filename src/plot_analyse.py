import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import scipy.spatial
import scipy.stats
import statsmodels.formula.api as smf
from utils.paths import path_expt, path_figures, path_metadata
from utils.tasks import (clutter, difficulty, scale, similarity_cnn,
    similarity_human, similarity_semantic, names, property_names, wnids,
    compute_task_properties)

mpl.rcParams['font.family'] = 'Fira Sans'
format_1dp = mpl.ticker.FormatStrFormatter('%.1f')
blue = '#2F80ED'
property_names = (
    'Clutter', 'Difficulty', 'Scale',
    'CNN similarity', 'Human similarity', 'Semantic similarity')
pad = lambda low, high: (low-((high-low)/20), high+((high-low)/20))
x_ticks = ((0, 5), (0, 0.9), (0, 1), (0, 0.8), (0, 0.3), (0, 1))
x_lims = tuple([pad(low, high) for low, high in x_ticks])
y_ticks = (0, 0.7)
y_lims = pad(*y_ticks)

# Show the distributions of clutter and scale within randomly selected classes
df_clutter = pd.read_csv(path_metadata/'imagenet_image_clutter.csv')
df_scale = pd.read_csv(path_metadata/'imagenet_image_scale.csv')
df_clutter['wnid'] = df_clutter['filepath'].str.split('/', expand=True)[0]
df_scale['wnid'] = df_scale['filepath'].str.split('/', expand=True)[0]
for df, property_name in zip((df_clutter, df_scale), ('clutter', 'scale')):
    n_rows, n_cols = 3, 5
    figure, axes = plt.subplots(
        n_rows, n_cols, sharex=True, sharey=False, figsize=(5, 2))
    x_min = round(min(df[property_name]))
    x_max = round(max(df[property_name]))
    x_step = (x_max - x_min) / 10
    bins = np.arange(x_min, x_max+x_step, x_step)
    np.random.seed(0)
    for i0, wnid in enumerate(np.random.choice(wnids, n_rows*n_cols)):
        scores = df[df['wnid'] == wnid][property_name]
        i, j = np.unravel_index(i0, (n_rows, n_cols))
        axes[i, j].hist(scores, color=blue, alpha=1, density=True, bins=bins)
        axes[i, j].set_xticks(())
        axes[i, j].set_yticks(())
    axes[-1, n_cols//2].set_xlabel(property_name.capitalize(), labelpad=10)
    axes[n_rows//2, 0].set_ylabel('Probability density', labelpad=10)
    figure.tight_layout(pad=0.1)
    path_save = path_figures/f'within_class_{property_name}.pdf'
    figure.savefig(path_save, bbox_inches='tight')

# Show the properties of (1) all candidate task sets and (2) the task sets
# selected for the experiment
tasks_all = [[i, j] for i in range(1000) for j in range(i+1, 1000)]
tasks_all = np.array(tasks_all)
properties_all = [compute_task_properties(i, j) for i, j in tasks_all]
properties_all = np.array(properties_all)
tasks_expt_binary = np.loadtxt(path_expt/'tasks.txt')[1:]
tasks_expt_inds = [np.flatnonzero(task) for task in tasks_expt_binary]
properties_expt = [compute_task_properties(i, j) for i, j in tasks_expt_inds]
properties_expt = np.array(properties_expt)
n = properties_all.shape[1] - 1
figure, axes = plt.subplots(n, n, figsize=(7, 7), sharex='col', sharey='row')
for i in range(n):
    for j in range(n):
        if i >= j:
            correlation, _ = scipy.stats.spearmanr(
                properties_all[:, i+1], properties_all[:, j])
            axes[i, j].plot(
                properties_all[:, j], properties_all[:, i+1], 'o',
                color='grey', markersize=1, alpha=0.2,
                markeredgewidth=0, rasterized=True)
            axes[i, j].plot(
                properties_expt[:, j], properties_expt[:, i+1], 'o',
                markersize=1, markeredgecolor=blue, markeredgewidth=0.2,
                fillstyle='none', rasterized=True)
            axes[i, j].text(
                0.92, 0.92, f'{correlation:.2f}',
                transform=axes[i, j].transAxes, ha='right', va='top',
                bbox=dict(facecolor='w', edgecolor='none', alpha=0.5))
            if j == 0:
                axes[i, j].set_ylabel(property_names[i+1])
                axes[i, j].set_ylim(x_lims[i+1])
                axes[i, j].set_yticks(x_ticks[i+1])
                axes[i, j].yaxis.set_major_formatter(format_1dp)
            if i == n - 1:
                axes[i, j].set_xlabel(property_names[j])
                axes[i, j].set_xlim(x_lims[j])
                axes[i, j].set_xticks(x_ticks[j])
                axes[i, j].xaxis.set_major_formatter(format_1dp)
        else:
            axes[i, j].set_axis_off()
figure.tight_layout(w_pad=0.6, h_pad=1)
path_save = path_figures/'task_set_property_scatter.pdf'
figure.savefig(path_save, dpi=600, bbox_inches='tight')

# Show the main experimental results: the perceptual boost of attention as a
# function of the six task-set properties
baseline = np.loadtxt(path_expt/'results.txt')[0]
results = np.loadtxt(path_expt/'results.txt')[1:]
acc_change = results - baseline
acc_change_in = []
for i, inds in enumerate(tasks_expt_inds):
    acc_change_in.append(np.mean(acc_change[i, inds]))
acc_change_in = np.array(acc_change_in)
x = properties_expt
y = acc_change_in
mins_maxes = np.stack((np.min(x, axis=0), np.max(x, axis=0)), axis=1)
means = np.mean(x, axis=0)
x_test = []
for i in range(mins_maxes.shape[0]):
    point_min, point_max = [], []
    for j in range(len(mins_maxes)):
        if i == j:
            point_min.append(mins_maxes[i, 0])
            point_max.append(mins_maxes[i, 1])
        else:
            point_min.append(means[j])
            point_max.append(means[j])
    x_test.append(np.array([point_min, point_max]))
model = sklearn.linear_model.LinearRegression().fit(x, y)
y_test = [model.predict(x_i) for x_i in x_test]
figure, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3), sharey=True)
for i in range(6):
    axes[i//3, i%3].plot(
        x[:, i], y, '.', color='grey', alpha=0.1,
        markersize=5, markeredgewidth=0)
    axes[i//3, i%3].plot(x_test[i][:, i], y_test[i], color=blue, linewidth=2)
    axes[i//3, i%3].set_xlim(x_lims[i])
    axes[i//3, i%3].set_xticks(x_ticks[i])
    axes[i//3, i%3].set_xlabel(property_names[i])
    axes[i//3, i%3].xaxis.set_major_formatter(format_1dp)
axes[0, 0].set_ylim(y_lims)
axes[0, 0].set_yticks(y_ticks)
axes[0, 0].set_ylabel('Accuracy change')
axes[1, 0].set_ylabel('Accuracy change')
figure.tight_layout(w_pad=0.5, h_pad=1.5)
figure.savefig(path_figures/'accuracy_change.pdf', bbox_inches='tight')

# Analyse the main experimental results using least-squares linear regression
x = properties_expt
x_std = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
terms = [name.replace(' ', '_') for name in property_names]
formula = 'Acc_change ~ ' + ' + '.join(terms)
df_nonstd = pd.DataFrame(x, columns=terms)
df_nonstd.insert(0, 'Acc_change', y)
model_nonstd = smf.ols(formula=formula, data=df_nonstd)
results_nonstd = model_nonstd.fit()
df_std = pd.DataFrame(x_std, columns=terms)
df_std.insert(0, 'Acc_change', y)
model_std = smf.ols(formula=formula, data=df_std)
results_std = model_std.fit()
df_results = pd.DataFrame({
    'i':range(len(results_nonstd.params)),
    'beta_i': np.around(list(results_nonstd.params), 3),
    'beta_hat_i': np.around(list(results_std.params), 3),
    'stderr_beta_i': np.around(list(results_nonstd.HC0_se), 3),
    'pvalue_i':np.around(list(results_nonstd.pvalues), 6),
    'description':(('Intercept',) + property_names)})
df_results.to_csv(
    path_figures/'accuracy_change.csv',
    float_format='%.3f',
    index=False,
    columns=('i', 'beta_i', 'beta_hat_i', 'stderr_beta_i', 'description'))

# Inspect the learnt attention weights to explain the importance of task-set
# difficulty in determining the perceptual boost of attention
weights0 = np.loadtxt(path_expt/'weights.txt')[0].reshape(1, -1)
weights = np.loadtxt(path_expt/'weights.txt')[1:]
distances = scipy.spatial.distance.cdist(weights0, weights)[0]
figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 2))
settings = dict(bins=20, density=True, alpha=0.5)
axes[0].hist(weights0.flatten(), color='grey', label='Task-generic', **settings)
axes[0].hist(weights.flatten(), color=blue, label='Task-specific', **settings)
axes[0].set_xlabel('Weight magnitude')
axes[0].set_ylabel('Probability density')
axes[0].set_yticks((0, 1, 2))
axes[0].legend()
axes[0].set_aspect(1.7)
plot = axes[1].scatter(distances, properties_expt[:, 1], s=5, c=acc_change_in)
x_label = 'Distance between task-generic and\n task-specific attention weights'
axes[1].set_xlabel(x_label)
axes[1].set_ylabel('Difficulty')
axes[1].set_yticks((0, 0.4, 0.8))
cbar = plt.colorbar(plot, aspect=15, ticks=(0.2, 0.4, 0.6))
cbar.outline.set_visible(False)
cbar.set_label('Accuracy change', rotation=270, labelpad=15)
figure.tight_layout(w_pad=5)
figure.savefig(path_figures/'attention_weights.pdf', bbox_inches='tight')
