import matplotlib as mpl


# source: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams
# setting the plotting parameters
def setPltParams():
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 2
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9


# removing x- and y-labels in NxN subplots
def setAxVisibility(axes):
    if axes.ndim > 1:
        for elem in axes[-1, 1:]:
          elem.get_yaxis().set_visible(False)
        for elem in axes[:-1, 0]:
          elem.get_xaxis().set_visible(False)
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                if i<axes.shape[0]-1 and j>0:
                    axes[i,j].get_xaxis().set_visible(False)
                    axes[i,j].get_yaxis().set_visible(False)
