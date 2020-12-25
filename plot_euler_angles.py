import numpy as np
import os
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math as m


def Rx(theta):
    return np.matrix([
                     [1, 0, 0],
                     [0, m.cos(theta), -m.sin(theta)],
                     [0, m.sin(theta), m.cos(theta)]
                    ])
  
def Ry(theta):
    return np.matrix([
                     [m.cos(theta), 0, m.sin(theta)],
                     [0, 1, 0],
                     [-m.sin(theta), 0, m.cos(theta)]
                    ])
  
def Rz(theta):
    return np.matrix([
                     [m.cos(theta), -m.sin(theta), 0],
                     [m.sin(theta), m.cos(theta) , 0],
                     [0, 0, 1]
                    ])

def rotation(yaw, pitch, roll, vector):
    R = Rz(roll) * Ry(pitch) * Rx(yaw)
    return R * vector

def rotation_of_xyz(yaw, pitch, roll, 
                    x=np.array([[1],[0],[0]]), 
                    y=np.array([[0],[1],[0]]), 
                    z=np.array([[0],[0],[1]])):
    yaw = m.radians(yaw)
    pitch = m.radians(pitch)
    roll = m.radians(roll)
    x_new = rotation(yaw, pitch, roll, x)
    y_new = rotation(yaw, pitch, roll, y)
    z_new = rotation(yaw, pitch, roll, z)
    return np.array(x_new), np.array(y_new), np.array(z_new)

def vec_to_line(vector=None):
    x, y, z = np.hstack((np.zeros(shape=(3, 1), dtype=float), vector))
    return x, y, z

def plot_euler_angles(pitch, yaw, roll, path_to_image):
    # Set up a figure twice as tall as it is wide
    fig = plt.figure(figsize=plt.figaspect(2.))
    ax = fig.add_subplot(2, 1, 2, projection='3d')

    # Get current rotation angle
    xline = [0, -1]
    yline = [0, 0]
    zline = [0, 0]
    ax.plot3D(xline, yline, zline, 'red', marker='o', linestyle='dashed')  # forward
    xline = [0, 0]
    yline = [0, 1]
    zline = [0, 0]
    ax.plot3D(xline, yline, zline, 'blue', marker='o', linestyle='dashed')  # left
    xline = [0, 0]
    yline = [0, 0]
    zline = [0, 1]
    ax.plot3D(xline, yline, zline, 'green', marker='o', linestyle='dashed')  # up
    x_new, y_new, z_new = rotation_of_xyz(yaw=yaw, pitch=pitch, roll=roll)
    x_new = -x_new
    x_line = vec_to_line(x_new)
    y_line = vec_to_line(y_new)
    z_line = vec_to_line(z_new)

    # Get current rotation angle
    ax.plot3D(x_line[0], x_line[1], x_line[2], 'red')  # forward

    ax.plot3D(y_line[0], y_line[1], y_line[2], 'blue')  # left
    ax.plot3D(z_line[0], z_line[1], z_line[2], 'green')  # up

    # Set rotation angle to 30 degrees
    ax.view_init(azim=160, elev=0)
    ax.axis('off')

    # Show the actual image
    ax = fig.add_subplot(2, 1, 1)
    img = mpimg.imread(os.path.join(os.getcwd(), path_to_image))
    ax.imshow(img)
    ax.set_ylabel('Damped oscillation')
    plt.show()


if __name__ == '__main__':
    plot_euler_angles(27.02, -28.71, -3.04, './data/samples/driver.jpg')