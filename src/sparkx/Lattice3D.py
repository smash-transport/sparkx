#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
import numpy as np
import warnings
from scipy.interpolate import interpn
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

class Lattice3D:
    """
    Represents a 3D lattice with evenly spaced points.

    Parameters
    ----------
    x_min : float
        The minimum value along the x-axis.
    x_max : float
        The maximum value along the x-axis.
    y_min : float
        The minimum value along the y-axis.
    y_max : float
        The maximum value along the y-axis.
    z_min : float
        The minimum value along the z-axis.
    z_max : float
        The maximum value along the z-axis.
    num_points_x : int
        The number of points along the x-axis.
    num_points_y : int
        The number of points along the y-axis.
    num_points_z : int
        The number of points along the z-axis.
    n_sigma_x : float
        Number of sigmas to consider in smearing in x direction. Default is 3.
    n_sigma_y : float
        Number of sigmas to consider in smearing in y direction. Default is 3.
    n_sigma_z : float
        Number of sigmas to consider in smearing in z direction. Default is 3.

    Attributes
    ----------
    x_min_ : float
        The minimum value along the x-axis.
    x_max_ : float
        The maximum value along the x-axis.
    y_min_ : float
        The minimum value along the y-axis.
    y_max_ : float
        The maximum value along the y-axis.
    z_min_ : float
        The minimum value along the z-axis.
    z_max_ : float
        The maximum value along the z-axis.
    num_points_x_ : int
        The number of points along the x-axis.
    num_points_y_ : int
        The number of points along the y-axis.
    num_points_z_ : int
        The number of points along the z-axis.
    spacing_x_ : float
        The spacing between points along the x-axis.
    spacing_y_ : float
        The spacing between points along the y-axis.
    spacing_z_ : float
        The spacing between points along the z-axis.
    density_x_ : float
        The density of points along the x-axis.
    density_y_ : float
        The density of points along the y-axis.
    density_z_ : float
        The density of points along the z-axis.
    cell_volume_ : float
        The volume of each cell in the lattice.
    x_values_ : numpy.ndarray
        The array of x-axis values.
    y_values_ : numpy.ndarray
        The array of y-axis values.
    z_values_ : numpy.ndarray
        The array of z-axis values.
    grid_ : numpy.ndarray
        The 3D grid containing the values at each lattice point.

    Methods
    -------
    set_value_by_index:
        Set the value at the specified indices in the grid.
    get_value_by_index:
        Get the value at the specified indices in the grid.
    set_value:
        Set the value at the specified coordinates in the lattice.
    set_value_nearest_neighbor:
        Set the value at the nearest neighbor of the specified coordinates 
        within the lattice.
    get_value:
        Get the value at the specified coordinates in the lattice.
    get_value_nearest_neighbor:
        Get the value of the nearest neighbor at the specified coordinates 
        within the lattice.
    get_coordinates:
        Get the coordinates corresponding to the given indices.
    find_closest_indices:
        Find the closest indices in the lattice for the given coordinates.
    interpolate_value:
        Interpolate the value at the specified position using trilinear interpolation.
    average:
        Compute the element-wise average of multiple Lattice3D objects.
    rescale:
        Rescale the values of the lattice by a specified factor.
    save_to_csv:
        Save the lattice data, including metadata, to a CSV file.
    load_from_csv:
        Load lattice data, including metadata, from a CSV file.
    visualize:
        Visualize the lattice data in a 3D plot.
    extract_slice:
        Extract a 2D slice from the lattice along the specified axis at the
        given index.
    save_slice_to_csv:
        Save a 2D slice from the lattice along the specified axis and index to
        a CSV file.
    interpolate_to_lattice:
        Interpolate the current lattice data to a new lattice with the specified
        number of points along each axis.
    interpolate_to_lattice_new_extent:
        Interpolate the current lattice data to a new lattice with the specified
        number of points and extent.
    reset:
        Reset the values of all grid points in the lattice to zero.
    add_same_spaced_grid:
        Add the values from another grid with the same spacing.
    add_particle_data:
        Add particle data to the lattice.

    """
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, num_points_x, num_points_y, num_points_z, n_sigma_x=None, n_sigma_y=None, n_sigma_z=None):
        self.x_min_ = x_min
        self.x_max_ = x_max
        self.y_min_ = y_min
        self.y_max_ = y_max
        self.z_min_ = z_min
        self.z_max_ = z_max
        self.num_points_x_ = num_points_x
        self.num_points_y_ = num_points_y
        self.num_points_z_ = num_points_z
        self.cell_volume_ = abs((x_max-x_min)*(y_max-y_min)*(z_max-z_min)/(num_points_x*num_points_y*num_points_z))

        self.x_values_ = np.linspace(x_min, x_max, num_points_x)
        self.y_values_ = np.linspace(y_min, y_max, num_points_y)
        self.z_values_ = np.linspace(z_min, z_max, num_points_z)

        self.grid_ = np.zeros((num_points_x, num_points_y, num_points_z))

        self.n_sigma_x_ = float(n_sigma_x) if n_sigma_x is not None else 3
        self.n_sigma_y_ = float(n_sigma_y) if n_sigma_y is not None else 3
        self.n_sigma_z_ = float(n_sigma_z) if n_sigma_z is not None else 3

        self.spacing_x_ = self.x_values_[1]-self.x_values_[0] if num_points_x >1 else  None
        self.spacing_y_ = self.y_values_[1]-self.y_values_[0] if num_points_y >1 else  None
        self.spacing_z_ = self.z_values_[1]-self.z_values_[0] if num_points_z >1 else  None
        self.density_x_ = (self.x_max_-self.x_min_)/self.num_points_x_
        self.density_y_ = (self.y_max_-self.y_min_)/self.num_points_y_
        self.density_z_ = (self.z_max_-self.z_min_)/self.num_points_z_

    def __is_valid_index(self, i, j, k):
        """
        Check if the given indices (i, j, k) are valid within the defined bounds.

        Parameters
        ----------
        i : int
            The index along the x-axis.
        j : int
            The index along the y-axis.
        k : int
            The index along the z-axis.

        Returns
        -------
        bool
            True if the indices are valid, False otherwise.
        """
        return (0 <= i < self.num_points_x_) and \
               (0 <= j < self.num_points_y_) and \
               (0 <= k < self.num_points_z_)

    def set_value_by_index(self, i, j, k, value):
        """
        Set the value at the specified indices (i, j, k) in the grid.

        Parameters
        ----------
        i : int
            The index along the x-axis.
        j : int
            The index along the y-axis.
        k : int
            The index along the z-axis.
        value : int or float
            The value to set at the specified indices.

        Returns
        -------
        None
        """
        if not self.__is_valid_index(i, j, k):
            warnings.warn("Provided indices are outside the lattice range.")
        else:
            self.grid_[i, j, k] = value

    def get_value_by_index(self, i, j, k):
        """
        Get the value at the specified indices (i, j, k) in the grid.

        Parameters
        ----------
        i : int
            The index along the x-axis.
        j : int
            The index along the y-axis.
        k : int
            The index along the z-axis.

        Returns
        -------
        int or float or None
            The value at the specified indices if the indices are valid, otherwise None.
        """
        if not self.__is_valid_index(i, j, k):
            warnings.warn("Provided indices are outside the lattice range.")
            return None
        else:
            return self.grid_[i, j, k]

    def __get_index(self, value, values, num_points):
        """
        Get the index corresponding to the given value within a specified range.

        Parameters
        ----------
        value : int or float
            The value for which the index is to be determined.
        values : list or numpy.ndarray
            The list or array containing the range of values, sorted ascending.
        num_points : int
            The number of points in the range.

        Returns
        -------
        int
            The index corresponding to the given value within the specified range.

        Raises
        ------
        ValueError
            If the value is outside the specified range.
        """
        if value < values[0] or value > values[-1]:
            raise ValueError("Value is outside the specified range.")

        index = np.searchsorted(values, value, side='right')
        if index == 0:
            index += 1

        return index - 1
    
    def __get_index_nearest_neighbor(self, value, values):
        """
        Get the index corresponding to the nearest neighbor of a given value 
        within a specified range.

        Parameters
        ----------
        value : int or float
            The value for which the index is to be determined.
        values : list or numpy.ndarray
            The list or array containing the range of values, sorted ascending.

        Returns
        -------
        int
            The index corresponding to the nearest neighbor of a given value 
            within the specified range.

        Raises
        ------
        ValueError
            If the value is outside the specified range.
        """
        if value < values[0] or value > values[-1]:
            raise ValueError("Value is outside the specified range.")

        index = np.abs(value - np.array(values)).argmin()

        return index

    def __get_indices(self, x, y, z):
        """
        Get the indices corresponding to the given coordinates within the lattice.

        Parameters
        ----------
        x : int or float
            The x-coordinate for which the index is to be determined.
        y : int or float
            The y-coordinate for which the index is to be determined.
        z : int or float
            The z-coordinate for which the index is to be determined.

        Returns
        -------
        tuple
            A tuple containing the indices (i, j, k) corresponding to the given
            coordinates.

        Raises
        ------
        ValueError
            If any of the coordinates are outside the specified ranges.
        """
        i = self.__get_index(x, self.x_values_, self.num_points_x_)
        j = self.__get_index(y, self.y_values_, self.num_points_y_)
        k = self.__get_index(z, self.z_values_, self.num_points_z_)
        return i, j, k
    
    def __get_indices_nearest_neighbor(self, x, y, z):
        """
        Get the indices corresponding to the nearest neighbor 
        given coordinates within the lattice.

        Parameters
        ----------
        x : int or float
            The x-coordinate for which the index is to be determined.
        y : int or float
            The y-coordinate for which the index is to be determined.
        z : int or float
            The z-coordinate for which the index is to be determined.

        Returns
        -------
        tuple
            A tuple containing the indices (i, j, k) of the nearest neighbor
            corresponding to the given coordinates.

        Raises
        ------
        ValueError
            If any of the coordinates are outside the specified ranges.
        """
        i = self.__get_index_nearest_neighbor(x, self.x_values_)
        j = self.__get_index_nearest_neighbor(y, self.y_values_)
        k = self.__get_index_nearest_neighbor(z, self.z_values_)
        return i, j, k

    def set_value(self, x, y, z, value):
        """
        Set the value at the specified coordinates within the lattice.

        Parameters
        ----------
        x : int or float
            The x-coordinate where the value is to be set.
        y : int or float
            The y-coordinate where the value is to be set.
        z : int or float
            The z-coordinate where the value is to be set.
        value : int or float
            The value to be set.

        Raises
        ------
        ValueError
            If any of the coordinates are outside their specified ranges.
        """
        i, j, k = self.__get_indices(x, y, z)
        self.set_value_by_index(i, j, k, value)

    def set_value_nearest_neighbor(self, x, y, z, value):
        """
        Set the value at the nearest neighbor of the 
        specified coordinates within the lattice.

        Parameters
        ----------
        x : int or float
            The x-coordinate where the value is to be set.
        y : int or float
            The y-coordinate where the value is to be set.
        z : int or float
            The z-coordinate where the value is to be set.
        value : int or float
            The value to be set.

        Raises
        ------
        ValueError
            If any of the coordinates are outside their specified ranges.
        """
        i, j, k = self.__get_indices_nearest_neighbor(x, y, z)
        self.set_value_by_index(i, j, k, value)

    def get_value(self, x, y, z):
        """
        Get the value at the specified coordinates within the lattice.

        Parameters
        ----------
        x : int or float
            The x-coordinate where the value is to be retrieved.
        y : int or float
            The y-coordinate where the value is to be retrieved.
        z : int or float
            The z-coordinate where the value is to be retrieved.

        Returns
        -------
        int or float or None
            The value at the specified coordinates. 
            
        Raises
        ------
        ValueError
            If any of the coordinates are outside their specified ranges.
        """
        i, j, k = self.__get_indices(x, y, z)
        return self.get_value_by_index(i, j, k)
    
    def get_value_nearest_neighbor(self, x, y, z):
        """
        Get the value of the nearest neighbor at the specified coordinates 
        within the lattice.

        Parameters
        ----------
        x : int or float
            The x-coordinate where the value is to be retrieved.
        y : int or float
            The y-coordinate where the value is to be retrieved.
        z : int or float
            The z-coordinate where the value is to be retrieved.

        Returns
        -------
        int or float or None
            The value at the specified coordinates.
        
        Raises
        ------
        ValueError
            If any of the coordinates are outside their specified ranges.
        """
        i, j, k = self.__get_indices_nearest_neighbor(x, y, z)
        return self.get_value_by_index(i, j, k)

    def __get_value(self, index, values, num_points):
        """
        Retrieve the value associated with the given index.

        Parameters
        ----------
        index : int
            The index of the value to retrieve.
        values : numpy.ndarray
            The array of values from which to retrieve the value.
        num_points : int
            The total number of points or elements in the array.

        Returns
        -------
        float
            The value associated with the given index.

        Raises
        ------
        ValueError
            If the index is outside the specified range.

        """
        if index < 0 or index >= num_points:
            raise ValueError("Index is outside the specified range.")
        return values[index]

    def get_coordinates(self, i, j, k):
        """
        Retrieve the coordinates associated with the given indices.

        Parameters
        ----------
        i : int
            The index along the x-axis.
        j : int
            The index along the y-axis.
        k : int
            The index along the z-axis.

        Returns
        -------
        tuple
            A tuple containing the x, y, and z coordinates corresponding to
            the given indices.

        Raises
        ------
        ValueError
            If any of the indices are outside the specified range.

        """
        x = self.__get_value(i, self.x_values_, self.num_points_x_)
        y = self.__get_value(j, self.y_values_, self.num_points_y_)
        z = self.__get_value(k, self.z_values_, self.num_points_z_)
        return x, y, z

    def __find_closest_index(self, value, values):
        """
        Find the index of the closest value to the given value in a list of values.

        Parameters
        ----------
        value : float
            The target value.
        values : list or numpy.ndarray
            The list or array of values to search.

        Returns
        -------
        int
            The index of the closest value in the list of values.

        """
        index = np.argmin(np.abs(values - value))
        return index
    
    def __is_within_range(self, x, y, z):
        """
        Check if the given coordinates are within the defined range.

        Parameters
        ----------
        x : float
            The x-coordinate.
        y : float
            The y-coordinate.
        z : float
            The z-coordinate.

        Returns
        -------
        bool
            True if the coordinates are within the defined range, False otherwise.

        """
        return (self.x_min_ <= x <= self.x_max_) and \
               (self.y_min_ <= y <= self.y_max_) and \
               (self.z_min_ <= z <= self.z_max_)

    def find_closest_indices(self, x, y, z):
        """
        Find the closest indices in the lattice corresponding to the given coordinates.

        Parameters
        ----------
        x : float
            The x-coordinate.
        y : float
            The y-coordinate.
        z : float
            The z-coordinate.

        Returns
        -------
        tuple
            A tuple of three integers representing the closest indices in the
            lattice for the given coordinates (x, y, z).

        """
        if not self.__is_within_range(x, y, z):
            warnings.warn("Provided position is outside the lattice range.")

        i = self.__find_closest_index(x, self.x_values_)
        j = self.__find_closest_index(y, self.y_values_)
        k = self.__find_closest_index(z, self.z_values_)
        return i, j, k

    def interpolate_value(self, x, y, z, method='nearest'):
        """
        Interpolate the value at the specified position using (up to) trilinear
        interpolation.

        Parameters
        ----------
        x : float
            The x-coordinate.
        y : float
            The y-coordinate.
        z : float
            The z-coordinate.
        method : str
            Scipy option for interpolation method. Default is 'nearest'.

        Returns
        -------
        float or None
            The interpolated value at the specified position. If the provided
            position is outside the lattice range, None is returned.

        """
        if not self.__is_within_range(x, y, z):
            warnings.warn("Provided position is outside the lattice range.")
            return None

        # Perform interpolation 
        xi = [x,y,z]
        return interpn((self.x_values_, self.y_values_, self.z_values_), self.grid_, xi, method=method)[0]

    def __operate_on_lattice(self, other, operation):
        """
        Apply a binary operation on two Lattice3D objects element-wise.

        Parameters
        ----------
        other : Lattice3D
            The other Lattice3D object to perform the operation with.
        operation : function
            The binary operation function to apply.

        Returns
        -------
        Lattice3D
            A new Lattice3D object containing the result of the element-wise
            operation.

        Raises
        ------
        TypeError
            If the operand `other` is not of type `Lattice3D`.
        ValueError
            If the lattices have different shapes.

        """
        if not isinstance(other, Lattice3D):
            raise TypeError("Unsupported operand type. The operand must be of type 'Lattice3D'.")

        if self.grid_.shape != other.grid_.shape:
            raise ValueError("The lattices must have the same shape.")

        result = Lattice3D(self.x_min_, self.x_max_, self.y_min_, self.y_max_, self.z_min_, self.z_max_,
                           self.num_points_x_, self.num_points_y_, self.num_points_z_)

        result.grid_ = operation(self.grid_, other.grid_)

        return result

    def __add__(self, other):
        """
        Add two Lattice3D objects element-wise.

        Parameters
        ----------
        other : Lattice3D
            The other Lattice3D object to add.

        Returns
        -------
        Lattice3D
            A new Lattice3D object containing the element-wise sum.

        """
        return self.__operate_on_lattice(other, lambda x, y: x + y)

    def __sub__(self, other):
        """
        Subtract two Lattice3D objects element-wise.

        Parameters
        ----------
        other : Lattice3D
            The other Lattice3D object to subtract.

        Returns
        -------
        Lattice3D
            A new Lattice3D object containing the element-wise difference.

        """
        return self.__operate_on_lattice(other, lambda x, y: x - y)

    def __mul__(self, other):
        """
        Multiply two Lattice3D objects element-wise.

        Parameters
        ----------
        other : Lattice3D
            The other Lattice3D object to multiply.

        Returns
        -------
        Lattice3D
            A new Lattice3D object containing the element-wise product.

        """
        return self.__operate_on_lattice(other, lambda x, y: x * y)

    def __truediv__(self, other):
        """
        Divide two Lattice3D objects element-wise.

        Parameters
        ----------
        other : Lattice3D
            The other Lattice3D object to divide.

        Returns
        -------
        Lattice3D
            A new Lattice3D object containing the element-wise division.

        Raises
        ------
        ZeroDivisionError
            If division by zero occurs during the element-wise division.

        """
        return self.__operate_on_lattice(other, lambda x, y: x / y)

    def average(self, *lattices):
        """
        Compute the average of multiple Lattice3D objects element-wise.

        Parameters
        ----------
        *lattices : Lattice3D
            Multiple Lattice3D objects to compute the average.

        Returns
        -------
        Lattice3D
            A new Lattice3D object containing the element-wise average.

        Raises
        ------
        TypeError
            If any of the operands are not of type 'Lattice3D'.
        ValueError
            If the lattices do not have the same shape.

        Examples
        --------

        If you want to average over 3 lattices, you can do the following:

        .. highlight:: python
        .. code-block:: python
            :linenos:

            >>> lattice1 = Lattice3D(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0,
            >>>    z_max=1, num_points_x=10, num_points_y=10, num_points_z=10)
            >>>
            >>> lattice2 = Lattice3D(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, 
            >>>    z_max=1, num_points_x=10, num_points_y=10, num_points_z=10)
            >>>
            >>> lattice3 = Lattice3D(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, 
            >>>    z_max=1, num_points_x=10, num_points_y=10, num_points_z=10)
            >>>
            >>> lattice1=lattice1.average(lattice2, lattice3)

        """
        all_lattices = [self] + list(lattices)

        for lattice in all_lattices:
            if not isinstance(lattice, Lattice3D):
                raise TypeError("Unsupported operand type. All operands must be of type 'Lattice3D'.")

            if self.grid_.shape != lattice.grid_.shape:
                raise ValueError("The lattices must have the same shape.")


        result = Lattice3D(self.x_min_, self.x_max_, self.y_min_, self.y_max_, self.z_min_, self.z_max_,
                           self.num_points_x_, self.num_points_y_, self.num_points_z_)

        result.grid_ = np.mean([lattice.grid_ for lattice in all_lattices], axis=0)

        return result

    def rescale(self, factor):
        """
        Rescale the values of the lattice by a specified factor.

        Parameters
        ----------
        factor : numeric
            The rescaling factor.

        Returns
        -------
        None

        """
        self.grid_ *= factor

    def save_to_csv(self, filename):
        """
        Save the lattice data, including metadata, to a CSV file.

        Parameters
        ----------
        filename : str
            The filename of the CSV file to save.

        Returns
        -------
        None

        """
        metadata = np.array([self.x_min_, self.x_max_, self.y_min_, self.y_max_, self.z_min_, self.z_max_,
                     self.num_points_x_, self.num_points_y_, self.num_points_z_])

        # Reshape metadata to have 1 row and as many columns as necessary
        metadata = metadata.reshape(1, -1)

        # Reshape the flattened grid to have 1 row and as many columns as necessary
        grid_flattened = self.grid_.flatten().reshape(1, -1)

        data = np.hstack((metadata, grid_flattened))
        np.savetxt(filename, data, delimiter=',')

    @classmethod
    def load_from_csv(cls, filename):
        """
        Load lattice data, including metadata, from a CSV file.

        Parameters
        ----------
        filename : str
            The filename of the CSV file to load.

        Returns
        -------
        lattice : Lattice3D
            The loaded Lattice3D object containing the data and metadata.

        """
        data = np.loadtxt(filename, delimiter=',')
        metadata = data[0:9]
        x_min, x_max, y_min, y_max, z_min, z_max, num_points_x, num_points_y, num_points_z = metadata

        lattice = Lattice3D(x_min, x_max, y_min, y_max, z_min, z_max, int(num_points_x), int(num_points_y), int(num_points_z))

        grid_data = data[9:]
        lattice.grid_ = grid_data.reshape(lattice.grid_.shape)

        return lattice

    def visualize(self):
        """
        Visualize the lattice data in a 3D plot.

        The lattice values are represented as a scatter plot in a 3D space.

        """
        # Generate grid coordinates
        X, Y, Z = np.meshgrid(self.x_values_, self.y_values_, self.z_values_)

        # Flatten the grid coordinates and values
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        values_flat = self.grid_.flatten()

        # Create a custom colormap where 0 values are white
        cmap = cm.get_cmap("PiYG").copy()
        cmap.set_bad(color='white')

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot the lattice points
        scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=values_flat, cmap=cmap)

        # Set plot labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lattice Data Visualization')

        # Create a color bar
        cbar = fig.colorbar(scatter)
        cbar.set_label('Values')

        # Show the plot
        plt.show()

    def extract_slice(self, axis, index):
        """
        Extract a 2D slice from the lattice along the specified axis at the
        given index.

        Parameters
        ----------
        axis : str
            The axis along which to extract the slice. Must be 'x', 'y', or 'z'.
        index : int
            The index of the slice along the specified axis.

        Returns
        -------
        A tuple containing the following:
        slice_data : ndarray
            The 2D slice data extracted from the lattice.
        slice_values : tuple of 2 ndarrays
            A tuple of coordinate lists containing the 2D coordinates of the slice.
        slice_label : str
            The label describing the slice plane.

        Raises
        ------
        ValueError
            If the `axis` parameter is invalid or the `index` is out of range.

        """
        if axis == 'x':
            if index < 0 or index >= self.num_points_x_:
                raise ValueError("Invalid index for the X-axis.")

            slice_data = self.grid_[index, :, :]
            slice_values = (self.y_values_,self.z_values_)
            slice_label = 'Y-Z Plane at X = {}'.format(self.x_values_[index])
        elif axis == 'y':
            if index < 0 or index >= self.num_points_y_:
                raise ValueError("Invalid index for the Y-axis.")

            slice_data = self.grid_[:, index, :]
            slice_values = (self.x_values_,self.z_values_)
            slice_label = 'X-Z Plane at Y = {}'.format(self.y_values_[index])
        elif axis == 'z':
            if index < 0 or index >= self.num_points_z_:
                raise ValueError("Invalid index for the Z-axis.")

            slice_data = self.grid_[:, :, index]
            slice_values = (self.x_values_,self.y_values_)
            slice_label = 'X-Y Plane at Z = {}'.format(self.z_values_[index])
        else:
            raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

        return slice_data, slice_values, slice_label

    def save_slice_to_csv(self, axis, index, filename):
        """
        Save a 2D slice from the lattice along the specified axis and index to
        a CSV file.

        Parameters
        ----------
        axis : str
            The axis along which to extract the slice. Must be 'x', 'y', or 'z'.
        index : int
            The index of the slice along the specified axis.
        filename : str
            The name of the CSV file to save the slice data.

        Raises
        ------
        ValueError
            If the `axis` parameter is invalid or the `index` is out of range.

        """
        if axis == 'x':
            if index < 0 or index >= self.num_points_x_:
                raise ValueError("Invalid index for the X-axis.")

            slice_data = self.grid_[index, :, :]
            header="#Slice in Y-Z Plane at X = {}. Ymin: {} Ymax: {} ".format(self.x_values_[index],self.y_min_,self.y_max_)+\
                "ny: {} Zmin: {} Zmax: {} nz:{} ".format(self.num_points_y_,self.z_min_,self.z_max_,self.num_points_z_)

        elif axis == 'y':
            if index < 0 or index >= self.num_points_y_:
                raise ValueError("Invalid index for the Y-axis.")

            slice_data = self.grid_[:, index, :]
            header="#Slice in X-Z Plane at Y = {}. Xmin: {} Xmax: {} ".format(self.y_values_[index],self.x_min_,self.x_max_)+\
                "nx: {} Zmin: {} Zmax: {} nz: {} ".format(self.num_points_x_,self.z_min_,self.z_max_,self.num_points_z_)
        elif axis == 'z':
            if index < 0 or index >= self.num_points_z_:
                raise ValueError("Invalid index for the Z-axis.")

            slice_data = self.grid_[:, :, index]
            header="#Slice in X-Y Plane at Z = {}. Xmin: {} Xmax: {} ".format(self.z_values_[index], self.x_min_,self.x_max_)+\
                "nx: {} Ymin: {} Ymax: {} ny: {} ".format(self.num_points_x_,self.y_min_,self.y_max_,self.num_points_y_)
        else:
            raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

        np.savetxt(filename, slice_data, delimiter=',', header=header, comments='')

    def interpolate_to_lattice(self, num_points_x, num_points_y, num_points_z):
        """
        Interpolate the current lattice data to a new lattice with the specified
        number of points along each axis.

        Parameters
        ----------
        num_points_x : int
            The number of points along the X-axis of the new lattice.
        num_points_y : int
            The number of points along the Y-axis of the new lattice.
        num_points_z : int
            The number of points along the Z-axis of the new lattice.

        Returns
        -------
        Lattice3D
            A new Lattice3D object containing the interpolated data.

        Notes
        -----
        This method performs spline interpolation to generate the values for
        each grid point of the new lattice. The interpolation is based on the
        current lattice data and the desired number of points along each axis.

        """
        # Create a new Lattice3D object with the desired number of points and resolution
        new_lattice = Lattice3D(self.x_min_, self.x_max_, self.y_min_, self.y_max_, self.z_min_, self.z_max_,
                                num_points_x, num_points_y, num_points_z)

        # Generate the new grid coordinates
        x_new = np.linspace(self.x_min_, self.x_max_, num_points_x)
        y_new = np.linspace(self.y_min_, self.y_max_, num_points_y)
        z_new = np.linspace(self.z_min_, self.z_max_, num_points_z)

        # Perform spline interpolation for each grid point of the new lattice
        for i, x in enumerate(x_new):
            for j, y in enumerate(y_new):
                for k, z in enumerate(z_new):
                    value = self.interpolate_value(x, y, z)
                    new_lattice.set_value_by_index(i, j, k, value)

        return new_lattice

    def interpolate_to_lattice_new_extent(self, num_points_x, num_points_y, num_points_z, x_min, x_max, y_min, y_max, z_min, z_max):
        """
        Interpolate the current lattice data to a new lattice with the specified
        number of points and extent.

        Parameters
        ----------
        num_points_x : int
            The number of points along the X-axis of the new lattice.
        num_points_y : int
            The number of points along the Y-axis of the new lattice.
        num_points_z : int
            The number of points along the Z-axis of the new lattice.
        x_min : float
            The minimum value of the X-axis for the new lattice extent.
        x_max : float
            The maximum value of the X-axis for the new lattice extent.
        y_min : float
            The minimum value of the Y-axis for the new lattice extent.
        y_max : float
            The maximum value of the Y-axis for the new lattice extent.
        z_min : float
            The minimum value of the Z-axis for the new lattice extent.
        z_max : float
            The maximum value of the Z-axis for the new lattice extent.

        Returns
        -------
        Lattice3D
            A new Lattice3D object containing the interpolated data.

        Notes
        -----
        This method performs spline interpolation to generate the values for
        each grid point of the new lattice. The interpolation is based on the
        current lattice data and the desired number of points and extent along
        each axis.

        """
        # Create a new Lattice3D object with the desired number of points and extent
        new_lattice = Lattice3D(x_min, x_max, y_min, y_max, z_min, z_max,
                                num_points_x, num_points_y, num_points_z)

        # Generate the new grid coordinates
        x_new = np.linspace(x_min, x_max, num_points_x)
        y_new = np.linspace(y_min, y_max, num_points_y)
        z_new = np.linspace(z_min, z_max, num_points_z)

        # Perform spline interpolation for each grid point of the new lattice
        for i, x in enumerate(x_new):
            for j, y in enumerate(y_new):
                for k, z in enumerate(z_new):
                    value = self.interpolate_value(x, y, z)
                    new_lattice.set_value_by_index(i, j, k, value)

        return new_lattice

    def reset(self):
        """
        Reset the values of all grid points in the lattice to zero.

        Notes
        -----
        This method iterates over all grid points in the lattice and sets their
        values to zero.
        """
        for i, j, k in np.ndindex(self.grid_.shape):
            self.grid_[i, j, k] = 0

    def add_same_spaced_grid(self, other, center_x, center_y, center_z):
        """
        Add the values of grid points of another lattice with same spacing.

        Parameters
        ----------
        other : Lattice3D
            The other Lattice3D object to add, which is expected to have the same spacing.
        center_x : float
            x-position in this grid at which the center of the other grid should be placed.
        center_y : float
            y-position in this grid at which the center of the other grid should be placed.
        center_z : float
            z-position in this grid at which the center of the other grid should be placed.

        Raises
        ------
        TypeError
            If the operand `other` is not of type `Lattice3D`.
        ValueError
            If `other` is of wrong spacing.

        """
        if not isinstance(other, Lattice3D):
            raise TypeError("Unsupported operand type. The operand must be of type 'Lattice3D'.")
        # Check if both lattices have the same spacing
        if ( ( other.spacing_x_ == None or abs(self.spacing_x_ - other.spacing_x_)<1e-3) and ( other.spacing_y_ == None or abs(self.spacing_y_ - other.spacing_y_)<1e-3 ) 
            and (other.spacing_z_ == None or abs(self.spacing_z_ - other.spacing_z_)<1e-3 )):
            for i, j, k in np.ndindex(other.grid_.shape):
                posx, posy, posz = other.get_coordinates(i,j,k)
                posx = posx + center_x
                posy = posy + center_y
                posz = posz + center_z
                if(posx<self.x_min_ or posx > self.x_max_ or posy<self.y_min_ or posy > self.y_max_ or posz<self.z_min_ or posz > self.z_max_):
                    continue
                else:
                    self.set_value_nearest_neighbor(posx,posy,posz,self.get_value_nearest_neighbor(posx,posy,posz) + other.grid_[i,j,k])

        else:
            raise ValueError("The provided lattices do not have identical spacing.")

    def add_particle_data(self, particle_data, sigma, quantity, kernel="covariant", add = False):
        """
        Add particle data to the lattice.

        Parameters
        ----------
        particle_data : list
            A list of Particle objects containing the particle data.
        sigma : float
            The standard deviation of the Gaussian kernel used for smearing the
            particle data.
        quantity : str
            The quantity of the particle data to be added. Supported values are
            'energy density', 'number', 'charge', and 'baryon number'.
        kernel : str, optional
            The type of kernel to use for smearing the particle data. Supported
            values are 'gaussian' and 'covariant'. The default is 'covariant'.
        add : bool, optional
            Specifies whether to add the particle data to the existing lattice
            values or replace them. If True, the particle data will be added to
            the existing lattice values. If False (default), the lattice values
            will be reset before adding the particle data.

        Raises
        ------
        ValueError
            If an unknown quantity is specified or if the particle data contains NaN values.

        Notes
        -----
        This method calculates the smearing of the particle data using a
        Gaussian kernel centered at each particle's coordinates. The smeared
        values are added to the corresponding grid points in the lattice.

        The Gaussian kernel is defined by the provided standard deviation
        `sigma`. The larger the `sigma` value, the smoother the smearing effect.

        The supported quantities for particle data are as follows:

        - 'energy_density': Uses the particle's energy (`E`) as the value to be added to the lattice.
        - 'number_density': Adds a value of 1.0 to the grid for each particle.
        - 'charge_density': Uses the particle's charge as the value to be added to the lattice.
        - 'baryon_density': Uses the particle's baryon number as the value to be added to the lattice.
        - 'strangeness_density': Uses the particle's strangeness as the value to be added to the lattice.

        """
        #delete old data?
        if not add:
             self.reset()
        for particle in particle_data:
            x = particle.x
            y = particle.y
            z = particle.z

            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                raise ValueError("Particle data contains NaN values.")

            if quantity == "energy_density":
                value = particle.E
            elif quantity == "number_density":
                value = 1.0
            elif quantity == "charge_density":
                value = particle.charge
            elif quantity == "baryon_density":
                value = particle.baryon_number
            elif quantity == "strangeness_density":
                value = particle.strangeness
            else:
                raise ValueError("Unknown quantity for lattice.")

            if np.isnan(value):
                raise ValueError("Particle data contains NaN values.")
            
            if(kernel == "gaussian"):
                # Calculate the Gaussian kernel centered at (x, y, z)
                kernel_value = multivariate_normal(mean=[x, y, z], cov=sigma**2 * np.eye(3))
            elif(kernel == "covariant"):
                if np.isnan(particle.px) or np.isnan(particle.py) or np.isnan(particle.py):
                    raise ValueError("Particle data contains NaN values.")
                kernel_value = multivariate_normal(mean=[0,0], cov=sigma**2 * np.eye(2))
            else:
                raise ValueError("Unknown kernel type for lattice.")

             # Determine the range of cells within the boundary
            range_x = self.n_sigma_x_ * sigma
            num_x = round(range_x / self.spacing_x_)
            range_x = num_x * self.spacing_x_
            
            range_y = self.n_sigma_y_ * sigma
            num_y = round(range_y / self.spacing_y_)
            range_y = num_y * self.spacing_y_
            
            range_z = self.n_sigma_z_ * sigma
            num_z = round(range_z / self.spacing_z_)
            range_z =num_z * self.spacing_z_ 
                
            norm = 0
            #we have now the right spacing, but the lattice points do not coincide
            temp_lattice=Lattice3D(-range_x, range_x, -range_y, range_y, -range_z, range_z, 2*num_x+1, 2*num_y+1, 2*num_z+1)

            for i in range(temp_lattice.num_points_x_):
                for j in range(temp_lattice.num_points_y_):
                    for k in range(temp_lattice.num_points_z_):
                        # Get the coordinates of the current grid point
                        xi, yj, zk = temp_lattice.get_coordinates(i, j, k)

                        # Calculate the value to add to the grid at (i, j, k)
                        if(kernel == "gaussian"):
                            smearing_factor = kernel_value.pdf([xi+x, yj+y, zk+z])
                        else:
                            diff_space = (xi)**2+(yj)**2+(zk)**2
                            gamma = np.sqrt(1+particle.p_abs()**2/particle.mass**2)
                            diff_velocity = (particle.px*(xi)+particle.py*(yj)+particle.pz*(zk))/(gamma*particle.mass)
                            smearing_factor = kernel_value.pdf([diff_space,diff_velocity])
                        norm += smearing_factor
                        value_to_add = value * smearing_factor / self.cell_volume_
                        
                        if np.isnan(value_to_add):
                            value_to_add=0.
                        # Add the value to the grid
                        temp_lattice.grid_[i, j, k] += value_to_add
                            
            for i in range(temp_lattice.num_points_x_):
                for j in range(temp_lattice.num_points_y_):
                    for k in range(temp_lattice.num_points_z_):
                        if abs(norm)>1e-15:
                            temp_lattice.grid_[i, j, k] /= norm
            xi ,yi ,zi = self.find_closest_indices(x, y, z)
            x_grid, y_grid, z_grid = self.get_coordinates(xi ,yi ,zi)
            self.add_same_spaced_grid(temp_lattice, x_grid, y_grid, z_grid)


def print_lattice(lattice):
    for i in range(lattice.num_points_x_):
        for j in range(lattice.num_points_y_):
            for k in range(lattice.num_points_z_):
                x, y, z = lattice.get_coordinates(i, j, k)
                value = lattice.get_value_by_index(i, j, k)
                print(f"Lattice point ({x}, {y}, {z}): {value}")
