import numpy as np
import warnings
from scipy.interpolate import interpn
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

class Lattice3D:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, num_points_x, num_points_y, num_points_z):
        self.x_min_ = x_min
        self.x_max_ = x_max
        self.y_min_ = y_min
        self.y_max_ = y_max
        self.z_min_ = z_min
        self.z_max_ = z_max
        self.num_points_x_ = num_points_x
        self.num_points_y_ = num_points_y
        self.num_points_z_ = num_points_z
        self.cell_volume = (x_max-x_min)*(y_max-y_min)*(z_max-z_min)/(num_points_x*num_points_y*num_points_z)

        self.x_values_ = np.linspace(x_min, x_max, num_points_x)
        self.y_values_ = np.linspace(y_min, y_max, num_points_y)
        self.z_values_ = np.linspace(z_min, z_max, num_points_z)

        self.grid_ = np.zeros((num_points_x, num_points_y, num_points_z))

    def __is_valid_index(self, i, j, k):
        return (0 <= i < self.num_points_x_) and \
               (0 <= j < self.num_points_y_) and \
               (0 <= k < self.num_points_z_)

    def set_value_by_index(self, i, j, k, value):
        if not self.__is_valid_index(i, j, k):
            warnings.warn("Provided indices are outside the lattice range.")
        else:
            self.grid_[i, j, k] = value

    def get_value_by_index(self, i, j, k):
        if not self.__is_valid_index(i, j, k):
            warnings.warn("Provided indices are outside the lattice range.")
            return None
        else:
            return self.grid_[i, j, k]

    def __get_index(self, value, values, num_points):
        if value < values[0] or value > values[-1]:
            raise ValueError("Value is outside the specified range.")

        index = np.searchsorted(values, value, side='right')
        if index == 0:
            index += 1
        elif index == num_points:
            index -= 1

        return index - 1

    def __get_indices(self, x, y, z):
        i = self.__get_index(x, self.x_values_, self.num_points_x_)
        j = self.__get_index(y, self.y_values_, self.num_points_y_)
        k = self.__get_index(z, self.z_values_, self.num_points_z_)
        return i, j, k

    def set_value(self, x, y, z, value):
        i, j, k = self.__get_indices(x, y, z)
        self.set_value_by_index(i, j, k, value)

    def get_value(self, x, y, z):
        i, j, k = self.__get_indices(x, y, z)
        return self.get_value_by_index(i, j, k)

    def __get_value(self, index, values, num_points):
        if index < 0 or index >= num_points:
            raise ValueError("Index is outside the specified range.")
        return values[index]

    def get_coordinates(self, i, j, k):
        x = self.__get_value(i, self.x_values_, self.num_points_x_)
        y = self.__get_value(j, self.y_values_, self.num_points_y_)
        z = self.__get_value(k, self.z_values_, self.num_points_z_)
        return x, y, z

    def __find_closest_index(self, value, values):
        index = np.argmin(np.abs(values - value))
        return index

    def __is_within_range(self, x, y, z):
        return (self.x_min_ <= x <= self.x_max_) and \
               (self.y_min_ <= y <= self.y_max_) and \
               (self.z_min_ <= z <= self.z_max_)

    def find_closest_indices(self, x, y, z):
        if not self.__is_within_range(x, y, z):
            warnings.warn("Provided position is outside the lattice range.")

        i = self.__find_closest_index(x, self.x_values_)
        j = self.__find_closest_index(y, self.y_values_)
        k = self.__find_closest_index(z, self.z_values_)
        return i, j, k
    
    def interpolate_value(self, x, y, z):
        if not self.__is_within_range(x, y, z):
            warnings.warn("Provided position is outside the lattice range.")
            return None

        # Check if the position falls exactly on a lattice point
        i, j, k = self.__get_indices(x, y, z)
        if (x == self.x_values_[i]) and (y == self.y_values_[j]) and (z == self.z_values_[k]):
            return self.grid_[i, j, k]

        # Perform trilinear interpolation
        xi = [x, y, z]
        return interpn((self.x_values_, self.y_values_, self.z_values_), self.grid_, xi)[0]
    
    def __operate_on_lattice(self, other, operation):
        if not isinstance(other, Lattice3D):
            raise TypeError("Unsupported operand type. The operand must be of type 'Lattice3D'.")

        if self.grid_.shape != other.grid_.shape:
            raise ValueError("The lattices must have the same shape.")

        result = Lattice3D(self.x_min_, self.x_max_, self.y_min_, self.y_max_, self.z_min_, self.z_max_,
                           self.num_points_x_, self.num_points_y_, self.num_points_z_)

        result.grid_ = operation(self.grid_, other.grid_)

        return result

    def __add__(self, other):
        return self.__operate_on_lattice(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self.__operate_on_lattice(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self.__operate_on_lattice(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self.__operate_on_lattice(other, lambda x, y: x / y)
    
    def average(self, *lattices):
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
        self.grid_ *= factor

    def save_to_csv(self, filename):
        metadata = np.array([self.x_min_, self.x_max_, self.y_min_, self.y_max_, self.z_min_, self.z_max_,
                             self.num_points_x_, self.num_points_y_, self.num_points_z_])

        data = np.vstack((metadata, self.grid_.flatten()))
        np.savetxt(filename, data, delimiter=',')

    def load_from_csv(filename):
        data = np.loadtxt(filename, delimiter=',')

        metadata = data[0]
        x_min, x_max, y_min, y_max, z_min, z_max, num_points_x, num_points_y, num_points_z = metadata

        lattice = Lattice3D(x_min, x_max, y_min, y_max, z_min, z_max, int(num_points_x), int(num_points_y), int(num_points_z))

        grid_data = data[1:]
        lattice.grid_ = grid_data.reshape(lattice.grid_.shape)

        return lattice

    def visualize(self):
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
        if axis == 'x':
            if index < 0 or index >= self.num_points_x_:
                raise ValueError("Invalid index for the X-axis.")

            slice_data = self.grid_[index, :, :]
            slice_values = self.y_values_
            slice_label = 'Y-Z Plane at X = {}'.format(self.x_values_[index])
        elif axis == 'y':
            if index < 0 or index >= self.num_points_y_:
                raise ValueError("Invalid index for the Y-axis.")

            slice_data = self.grid_[:, index, :]
            slice_values = self.x_values_
            slice_label = 'X-Z Plane at Y = {}'.format(self.y_values_[index])
        elif axis == 'z':
            if index < 0 or index >= self.num_points_z_:
                raise ValueError("Invalid index for the Z-axis.")

            slice_data = self.grid_[:, :, index]
            slice_values = self.x_values_
            slice_label = 'X-Y Plane at Z = {}'.format(self.z_values_[index])
        else:
            raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

        return slice_data, slice_values, slice_label

    def save_slice_to_csv(self, axis, index, filename):
        if axis == 'x':
            if index < 0 or index >= self.num_points_x_:
                raise ValueError("Invalid index for the X-axis.")

            slice_data = self.grid_[index, :, :]
            slice_values = self.y_values_
        elif axis == 'y':
            if index < 0 or index >= self.num_points_y_:
                raise ValueError("Invalid index for the Y-axis.")

            slice_data = self.grid_[:, index, :]
            slice_values = self.x_values_
        elif axis == 'z':
            if index < 0 or index >= self.num_points_z_:
                raise ValueError("Invalid index for the Z-axis.")

            slice_data = self.grid_[:, :, index]
            slice_values = self.x_values_
        else:
            raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

        np.savetxt(filename, slice_data, delimiter=',', header=','.join(map(str, slice_values)), comments='')

    def interpolate_to_lattice(self, num_points_x, num_points_y, num_points_z):
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
    
    def add_particle_data(self, particle_data, sigma, quantity):
        for particle in particle_data:
            x = particle.x
            y = particle.y
            z = particle.z
            
            if quantity == "energy density":
                value = particle.E
            else:
                raise ValueError("Unknown quantity for lattice.");

            # Calculate the Gaussian kernel centered at (x, y, z)
            kernel = multivariate_normal([x, y, z], cov=sigma**2 * np.eye(3))

            for i, j, k in np.ndindex(self.grid_.shape):
                # Get the coordinates of the current grid point
                xi, yj, zk = self.get_coordinates(i, j, k)

                # Calculate the value to add to the grid at (i, j, k)
                smearing_factor = kernel.pdf([xi, yj, zk])
                value_to_add = value * smearing_factor / self.cell_volume

                # Add the value to the grid
                self.grid_[i, j, k] += value_to_add

def print_lattice(lattice):
    for i in range(lattice.num_points_x_):
        for j in range(lattice.num_points_y_):
            for k in range(lattice.num_points_z_):
                x, y, z = lattice.get_coordinates(i, j, k)
                value = lattice.get_value_by_index(i, j, k)
                print(f"Lattice point ({x}, {y}, {z}): {value}")

"""
latt = Lattice3D(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, z_min=-5.0, z_max=5.0, num_points_x=5, num_points_y=5, num_points_z=5)

latt.set_value_by_index(0,0,0,42.42)
latt.set_value_by_index(3,3,3,-24)

print(latt.get_coordinates(0,0,0))
print(latt.get_value_by_index(0,0,0))

print(latt.find_closest_indices(-4.8,-4.8,-4.5))

latt.visualize()
print_lattice(latt)


# Extract a slice along the X-axis at index 5
slice_data, slice_values, slice_label = latt.extract_slice('x', 4)

# Plot the slice
plt.imshow(slice_data, extent=[slice_values.min(), slice_values.max(), latt.z_min_, latt.z_max_], origin='lower', cmap='jet')
plt.colorbar(label='Values')
plt.xlabel('Y')
plt.ylabel('Z')
plt.title(slice_label)
plt.show()

"""
