import matplotlib.pyplot as plt
import numpy as np


def visualize_dimensionality_reduction(
    transformation: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    targets: list,
    title: str | None = None,
    save_to: str | None = None,
) -> None:
    """Create a scatter visualisation for passed transformation.

    Creates a matplotlib scatter to visualise transformation of multidimensional
      data with colors equal to passed targets.

    Parameters
    ----------
    transformation : _type_
        A transformation of multidimensional data.
    targets : list
        A labels list.
    title : str | None, optional
        A title to add on visualization, by default None
    save_to : str | None, optional
        A path to location where visualization should be stored, by default None
    """
    # defuine colormap
    cmap = plt.cm.tab20b

    # create a scatter plot of the transformation output
    plt.scatter(
        transformation[:, 0],
        transformation[:, 1],
        c=np.array(targets).astype(int),
        cmap=cmap,
    )

    # save unique labels
    labels = np.unique(targets)

    # retrieve color used for each target
    norm = plt.Normalize(
        vmin=min(np.array(labels).astype(int)),
        vmax=max(np.array(labels).astype(int)),
    )
    rgba_values = cmap(norm(labels))

    # create a legend with the class labels and colors
    handles = [
        plt.scatter([], [], c=rgba, label=label)
        for rgba, label in zip(rgba_values, labels, strict=True)
    ]
    plt.legend(handles=handles, title="Classes")
    plt.axis("off")

    # add title
    if title:
        plt.title(title)

    # save image
    if save_to:
        plt.savefig(save_to)

    plt.show()
