import matplotlib.pyplot as plt 
import numpy as np


def visualize_dimensionality_reduction(
                                       transformation, 
                                       targets:list, 
                                       title:str|None=None, 
                                       save_to:str|None=None
                                       ) -> None:
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

    title: str, optional
        A title to add on vizualization.

    save_to: str, optional
        A path to location where vizualization should be stored
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

    # add title
    if title:
        plt.title(title)

    # save image
    if save_to:
        plt.savefig(save_to)
        
    plt.show()