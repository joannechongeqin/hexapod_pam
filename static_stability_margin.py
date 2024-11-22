# Static Stability Margin (SSM)

import torch
import matplotlib.pyplot as plt
from torchmin import minimize

def convex_hull_pam(points):
    """
    Computes the convex hull of 2D points using gift wrapping / jarvis march algorithm.
    
    :params torch.Tensor points: 2D points. Shape (batch_size, num_points, 2).

    :return: Convex hull vertices. Shape (batch_size, N_hull_points, 2).
    :rtype: torch.Tensor 
    """
    batch_size = points.shape[0]
    hull_batch = []

    for b in range(batch_size):
        batch_points = points[b]
        n = batch_points.shape[0] # num_points

        # Start with the leftmost point (guaranteed to be on hull)
        start_idx = torch.argmin(batch_points[:, 0])
        hull = [start_idx]
        current_idx = start_idx

        while True:
            next_idx = (current_idx + 1) % n

            for i in range(n):
                if i == current_idx:
                    continue
                # Compute cross product to find counter-clockwise turn
                a = batch_points[current_idx]
                b = batch_points[next_idx]
                c = batch_points[i]
                cross_product = torch.cross(
                    torch.cat([b - a, torch.tensor([0.0])]),
                    torch.cat([c - a, torch.tensor([0.0])]),
                    dim=-1
                )[-1]  # Z-component of cross product
                if cross_product < 0:  # Counter-clockwise turn
                    next_idx = i

            # If looped back to the start, exit
            if next_idx == start_idx:
                break

            hull.append(next_idx)
            current_idx = next_idx

        # Gather hull points for the batch
        hull_batch.append(batch_points[torch.tensor(hull)])
    
    # Stack the results across batches
    hull_tensor = torch.stack(hull_batch, dim=0)
    return hull_tensor


def point_to_edge_distances(point, hull):
    """
    Computes the perpendicular distances from a point to each edge of a convex hull polygon.
    
    :param torch.Tensor point: The interior point. Shape (batch_size, 2).
    :param torch.Tensor hull: Convex hull vertices. Shape (batch_size, N_hull_points, 2).
    
    :return: Perpendicular distances from the point to each edge of the convex hull. Shape (batch_size, N_hull_points).
    :rtype: torch.Tensor
    """
    batch_size = hull.shape[0]
    distances_batch = []

    for b in range(batch_size):
        batch_hull = hull[b]
        batch_point = point[b]

        # Repeat the first point at the end to close the polygon
        hull_extended = torch.cat([batch_hull, batch_hull[0].unsqueeze(0)], dim=0)

        # Compute all edges (vectors)
        edges = hull_extended[1:] - hull_extended[:-1]  # Shape: (N, 2)

        # Vector from the point to the start of each edge
        to_point = batch_point - hull_extended[:-1]  # Shape: (N, 2)

        # Compute the perpendicular distance to each edge
        # Cross product of edge and vector to point gives area of parallelogram
        cross_products = torch.abs(edges[:, 0] * to_point[:, 1] - edges[:, 1] * to_point[:, 0])

        # Normalize by edge length to get height (distance)
        edge_lengths = torch.norm(edges, dim=1)
        distances = cross_products / edge_lengths  # Shape: (N,)

        distances_batch.append(distances)

    # Stack the results across batches
    distances_tensor = torch.stack(distances_batch, dim=0)
    return distances_tensor


def point_in_hull(point, hull):
    """
    Checks whether a given point lies inside a convex hull.

    Uses the cross product method to determine if the point is inside the convex polygon. 
    If the point lies to the left of every edge of the polygon, it is considered inside the hull. 
    If it lies to the right of any edge, it is considered outside.

    :param torch.Tensor point: A 2D point to check if it lies inside the convex hull. Shape (2,).
    :param torch.Tensor hull: Vertices of the convex hull in counterclockwise order. Shape (num_vertices, 2).

    :return: `True` if the point lies inside the convex hull, `False` otherwise.
    :rtype: bool
    """
    num_edges = hull.shape[0]
    inside = True

    for i in range(num_edges):
        start = hull[i]
        end = hull[(i + 1) % num_edges]
        edge_vector = end - start
        to_point_vector = point - start

        # Compute cross product to determine if the point is to the left of the edge (counter-clockwise)
        cross_product = torch.cross(torch.cat([edge_vector, torch.tensor([0.0])]), torch.cat([to_point_vector, torch.tensor([0.0])]), dim=-1)[-1]
        
        if cross_product < 0:  # If the point is to the right of the edge (clockwise), it's outside
            inside = False
            break
    
    return inside


def static_stability_margin(points, base_pos):
    """
    Computes the static stability margin of the hexapod.
    
    Calculates the minimum distance from the base position to each edge of the convex hull formed by the legs' contact points.

    :param torch.Tensor points: 2D points representing the contact points of the hexapod legs. Shape (batch_size, num_points, 2).    
    :param torch.Tensor base_pos: Base position of the hexapod. Shape (batch_size, 2).
    
    :return: The static stability margin for each hexapod in the batch. Shape (batch_size,).
    :rtype: torch.Tensor
    """
    # Step 1: Compute the convex hull of the contact points
    hull_points = convex_hull_pam(points)
    
    # Step 2: Compute the minimum distance from the base to the convex hull's edges
    distances = point_to_edge_distances(base_pos, hull_points)
    
    # The stability margin is the minimum of these distances
    stability_margin = torch.min(distances, dim=1).values
    
    return stability_margin


# Visualize the convex hull and distances
def visualize_convex_hull(points, hull_points, interior_point, batch_idx=0):
    """
    Visualizes the convex hull, points, and distances for a particular batch.
    """
    # Get the points, hull, and interior point for the selected batch
    points_batch = points[batch_idx].detach().numpy()
    hull_batch = hull_points[batch_idx].detach().numpy()
    interior_point_batch = interior_point[batch_idx].detach().numpy()

    # Close the hull for plotting
    hull_closed = torch.cat([hull_points[batch_idx], hull_points[batch_idx][:1]], dim=0).detach().numpy()

    # Plot all points
    plt.figure(figsize=(8, 8))
    plt.scatter(points_batch[:, 0], points_batch[:, 1], label='Points', color='blue')
    plt.scatter(interior_point_batch[0], interior_point_batch[1], label='Interior Point', color='red', zorder=5)

    # Plot convex hull
    plt.plot(hull_closed[:, 0], hull_closed[:, 1], label='Convex Hull', color='green')

    # Plot distances
    for i in range(hull_batch.shape[0]):
        start = torch.tensor(hull_batch[i])  # Convert back to tensor for consistent computation
        end = torch.tensor(hull_batch[(i + 1) % hull_batch.shape[0]])
        edge_vector = end - start
        edge_vector /= torch.norm(edge_vector)  # Normalize the edge vector
        
        # Perpendicular distance vector
        to_edge_vector = torch.tensor(interior_point_batch) - start
        projection_length = torch.dot(to_edge_vector, edge_vector)
        projection = projection_length * edge_vector
        perpendicular_point = start + projection
        
        plt.plot(
            [interior_point_batch[0], perpendicular_point[0].item()],
            [interior_point_batch[1], perpendicular_point[1].item()],
            linestyle='--', color='orange'
        )

    plt.legend()
    plt.grid(True)
    plt.title(f"Convex Hull and Distances (Batch {batch_idx})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()


def ssm_cost_function(body_xys):
    ssm = static_stability_margin(eef_support_xy_pos, body_xys)
    cost = ssm.mean()
    
    # Check if each body_xys point is inside the support polygon
    for i in range(body_xys.shape[0]):
        body_point = body_xys[i]
        if not point_in_hull(body_point, hull_points2[i]):
            cost -= 1e6  # Apply penalty for points outside the polygon

    return - cost


if __name__=='__main__':

    # --- TEST 1 ---
    # points = torch.tensor([
    #     [
    #         [0.4412, 0.2555],
    #         [0.4411, -0.1315],
    #         [-0.3564, 0.3292],
    #         [-0.3565, -0.2049],
    #         [0.0, 0.0]  # A point that is inside the hull
    #     ],
    #     [
    #         [0.5, 0.5],
    #         [0.6, -0.2],
    #         [-0.1, 0.7],
    #         [-0.2, -0.5],
    #         [0.2, 0.0]  # A point that is inside the second hull
    #     ]
    # ])

    # # Compute convex hull for the batch
    # hull_points = convex_hull_pam(points)

    # # Example interior points (one per batch)
    # interior_points = torch.tensor([
    #     [0.0, 0.0],  # Inside the first convex hull
    #     [0.1, 0.1]   # Inside the second convex hull
    # ])

    # # Compute distances from the interior point to the edges of the convex hull
    # distances = point_to_edge_distances(interior_points, hull_points)

    # print("Convex Hull Points:")
    # print(hull_points)
    # print("\nDistances to each edge of the convex hull:")
    # print(distances)
    # visualize_convex_hull(points, hull_points, interior_points, batch_idx=0)
    # visualize_convex_hull(points, hull_points, interior_points, batch_idx=1)
    # stability_margin_values = static_stability_margin(points, interior_points)


    # --- TEST 2 ---
    eef_support_xy_pos = torch.tensor([
        [[ 0.0104,  0.4236],
        [ 0.0103, -0.4166],
        [-0.4179,  0.2754],
        [-0.4180, -0.2683]],

        [[-0.0182,  0.4863],
        [-0.0184, -0.3715],
        [-0.4543,  0.3337],
        [-0.4543, -0.2188]]
    ])

    body_xys = torch.tensor([
        [-0.0321,  0.0035],
        [-0.0608,  0.0574]
    ])

    hull_points2 = convex_hull_pam(eef_support_xy_pos)
    distances_before = point_to_edge_distances(body_xys, hull_points2)
    print("\nDistances to each edge of the convex hull before:")
    print(distances_before)
    stability_margin_values = static_stability_margin(eef_support_xy_pos, body_xys)
    print("Static Stability Margin before:")
    print(stability_margin_values)
    visualize_convex_hull(eef_support_xy_pos, hull_points2, body_xys, batch_idx=0)
    visualize_convex_hull(eef_support_xy_pos, hull_points2, body_xys, batch_idx=1)

    res = minimize(
            ssm_cost_function, 
            body_xys, 
            method='l-bfgs', 
            options=dict(line_search='strong-wolfe'),
            max_iter=50,
            disp=2
            )
    print("res.success: ", res.success)
    print(res.x)

    optimized_base_pos = res.x

    distances_after = point_to_edge_distances(optimized_base_pos, hull_points2)
    print("\nDistances to each edge of the convex hull after:")
    print(distances_after)
    visualize_convex_hull(eef_support_xy_pos, hull_points2, optimized_base_pos, batch_idx=0)
    visualize_convex_hull(eef_support_xy_pos, hull_points2, optimized_base_pos, batch_idx=1)
    stability_margin_values = static_stability_margin(eef_support_xy_pos, optimized_base_pos)
    print("Static Stability Margin after:")
    print(stability_margin_values)
