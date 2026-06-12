import matplotlib.pyplot as plt 
import numpy as np


def visualize_dimensionality_reduction(transformation, targets, title=None):
    r"""
    Creates a scatter visualisation for
    passed transformation.

    Creates a matplotlib scatter to visualise
    transformation of multidimensional data
    with colors equal to passed targets.

    Parameters
    ----------
    transformation : Any
        A transformation of multidimensional
        data.

    targets: list
        A labels list.
    """
    # defuine colormap
    cmap = plt.cm.tab20b # noqa

    # create a scatter plot of the transformation output
    plt.scatter(transformation[:, 0], transformation[:, 1],
                c=np.array(targets).astype(int), cmap=cmap) 

    # save unique labels
    labels = np.unique(targets)

    # retrieve color used for each target
    norm = plt.Normalize(vmin=min(np.array(labels).astype(int)), vmax=max(np.array(labels).astype(int)))
    rgba_values = cmap(norm(labels))

    # create a legend with the class labels and colors
    handles = [plt.scatter([], [], c=rgba, label=label) for rgba, label in zip(rgba_values, labels)]
    plt.legend(handles=handles, title='Classes')
    plt.axis('off') 
    if title:
        plt.title(title)
    plt.show()