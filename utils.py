import numpy as np
import matplotlib.pyplot as plt



def generate_origin_vectors(number_of_points:int) -> tuple[np.ndarray, np.ndarray]:
    """ Generate origin vectors for a given number of points
    
    Args:
        number_of_points (int): Number of points
    
    Returns:
        np.ndarray: Origin vectors
    
    """

    return np.zeros((number_of_points, 1)), np.zeros((number_of_points, 1))


def plot_vectors(V:np.ndarray,ax :plt.Axes, colors :list[str] = None,title :str = ""):
    """ Plot vectors
    
    Args:
        V (np.ndarray): Vectors of shape (n, 2) each row is a vector to plot
        colors (list[str], optional): Colors. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "".
    """
    if colors is not None:
        if V.shape[0] != len(colors):
            raise ValueError("Number of colors must be equal to number of vectors")
    else:
        colors = ["blue"] * V.shape[0]
    
    if ax is None:
        raise ValueError("ax must not be None")
    
    x,y = generate_origin_vectors(V.shape[0])

    ax.quiver(x,y,V[:,0],V[:,1],color=colors,angles='xy', scale_units='xy', scale=1)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')  

    # Find xlim and ylim
    x_max = np.max(V[:,0])
    x_min = np.min(V[:,0])
    y_max = np.max(V[:,1])
    y_min = np.min(V[:,1])
    
    ax.set_xlim(x_min-0.5, x_max+0.5)
    ax.set_ylim(y_min-0.5, y_max+0.5)
    



def generate_vector_space(number_of_points:int) -> np.ndarray:
    """ Generate vector space for a given number of points
    
    Args:
        number_of_points (int): Number of points
    
    Returns:
        np.ndarray: Vector space (number_of_points*2, 2)
        First half of the vectors are positive y values and second half are negative y values
    
    """

    x_values = np.linspace(-1,1,number_of_points)
    y_values = np.sqrt(1-x_values**2)

    positive_coords = np.column_stack((x_values,y_values))
    negative_coords = np.column_stack((x_values,-y_values))

    return np.concatenate((positive_coords,negative_coords),axis=0)