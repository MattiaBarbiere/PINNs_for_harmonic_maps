"""
Embedding Layer Test Suite.

This module provides comprehensive testing functionality for the embedding layer
used in Physics-Informed Neural Networks (PINNs) for positional encoding.

The tests validate collision detection and functionality based on the algorithm
described in Kast et al. 2023, ensuring that the embedding layer provides
adequate separation between nearby points in the embedded space.

Classes:
    None (functional module)

Functions:
    test_embedding_layer: Main collision detection test
    test_embedding_functionality: Basic functionality test
    run_embedding_tests: Comprehensive test runner
"""

from typing import Tuple, List
import torch

from hmpinn.embedding import Embedding_layer


def test_embedding_layer(num_embeddings_per_dim: int = 2, 
                        mesh_resolution: int = 10) -> Tuple[bool, List[torch.Tensor]]:
    """
    Test the embedding layer for collisions using Kast et al. 2023 algorithm.
    
    This test creates a regular mesh of points and checks whether the embedding
    layer provides sufficient separation in the embedded space. A collision occurs
    when two points that are far apart in physical space are mapped close together
    in embedding space.
    
    Args:
        num_embeddings_per_dim: Number of frequency embeddings per spatial dimension.
        mesh_resolution: Resolution of the test mesh (points per dimension).
        
    Returns:
        Tuple[bool, List[torch.Tensor]]: A tuple containing:
            - bool: True if test passes (no collisions), False otherwise
            - List[torch.Tensor]: List of collision points (empty if no collisions)
    """
    # Create embedding layer
    embedding = Embedding_layer(num_embeddings_per_dim)
    
    # Set tolerances as suggested in Kast et al. 2023
    spatial_tolerance = 1.0 / mesh_resolution
    embedding_tolerance = 0.1 * spatial_tolerance
    
    # Generate uniform test mesh
    x_coords = torch.linspace(0.1, 0.9, mesh_resolution)
    y_coords = torch.linspace(0.1, 0.9, mesh_resolution)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    mesh_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Check for collisions between all pairs of points
    collision_points = []
    num_points = len(mesh_points)
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            point_i = mesh_points[i].reshape(1, 2)
            point_j = mesh_points[j].reshape(1, 2)
            
            # Compute embeddings for both points
            embedding_i = embedding(point_i)
            embedding_j = embedding(point_j)
            
            # Check collision condition
            embedding_distance = torch.norm(embedding_i - embedding_j)
            spatial_distance = torch.norm(mesh_points[i] - mesh_points[j])
            
            # Collision detected if points are close in embedding but far in space
            if (embedding_distance < embedding_tolerance and 
                spatial_distance > spatial_tolerance):
                collision_points.extend([mesh_points[i], mesh_points[j]])
                return False, collision_points
    
    return True, collision_points


def test_embedding_functionality() -> None:
    """
    Test basic embedding layer functionality and properties.
    
    Validates that the embedding layer correctly increases dimensionality,
    maintains batch size, and has properly initialized frequency parameters.
    
    Raises:
        AssertionError: If any functionality test fails.
    """
    # Test with sample points
    test_points = torch.tensor([[0.1, 0.1], [0.3, 0.5]])
    embedding = Embedding_layer(2)
    
    # Check that embedding produces output with correct batch size
    result = embedding(test_points)
    assert result.shape[0] == test_points.shape[0], \
        "Embedding output batch size mismatch"
    
    # Check that embedding increases dimensionality
    assert result.shape[1] > test_points.shape[1], \
        "Embedding should increase dimensionality"
    
    # Check that frequencies are properly initialized
    assert hasattr(embedding, 'frequencies'), \
        "Embedding should have frequencies attribute"
    assert embedding.frequencies.shape[0] > 0, \
        "Frequencies should not be empty"
    
    # Check that output is differentiable (has gradient information)
    test_points_grad = torch.tensor([[0.1, 0.1], [0.3, 0.5]], requires_grad=True)
    result_grad = embedding(test_points_grad)
    loss = result_grad.sum()
    loss.backward()
    assert test_points_grad.grad is not None, \
        "Embedding should preserve gradient computation"


def run_embedding_tests() -> None:
    """
    Run comprehensive embedding layer tests.
    
    Executes all embedding tests including collision detection and functionality
    validation. Provides detailed output about test progress and results.
    
    Raises:
        AssertionError: If any tests fail, with specific failure information.
    """
    print("Testing embedding layer collision detection...")
    test_passed, collisions = test_embedding_layer()
    
    if not test_passed:
        print(f"Collision test failed! Found {len(collisions)//2} collision pairs")
        raise AssertionError("Embedding layer collision test failed")
    
    print("Testing embedding functionality...")
    test_embedding_functionality()
    
    print("All embedding tests passed!")


if __name__ == "__main__":    
    # Demonstration of embedding layer usage
    print("Embedding Layer Demonstration:")
    print("=" * 40)
    test_points = torch.tensor([[0.1, 0.1], [0.3, 0.5]])
    embedding = Embedding_layer(2)
    
    print(f"Input points shape: {test_points.shape}")
    print(f"Input points:\n{test_points}")
    
    embedded_points = embedding(test_points)
    print(f"\nEmbedded points shape: {embedded_points.shape}")
    print(f"Embedded points:\n{embedded_points}")
    print(f"Frequencies shape: {embedding.frequencies.shape}")
    
    print("\n" + "=" * 40)
    print("Running tests...")
    
    # Run all tests
    run_embedding_tests()