import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Define the points
points = np.array([
    [910, 793],
    [910, 793],
    [909, 797],
    [908, 815],
    [908, 831],
    [908, 841],
    [931, 858],
    [952, 860],
    [955, 860]
])

# Compute the convex hull
hull = ConvexHull(points)

# Extract the vertices
hull_points = points[hull.vertices]

# Plot all points
plt.scatter(points[:, 0], points[:, 1], label='Points')

# Highlight the convex hull vertices
plt.scatter(hull_points[:, 0], hull_points[:, 1], color='red', label='Hull Vertices')

# Draw the convex hull
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

# Annotate points with their indices
for i, point in enumerate(points):
    plt.text(point[0], point[1], str(i), fontsize=12, ha='right')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Convex Hull of Points')
plt.legend()
plt.show()

