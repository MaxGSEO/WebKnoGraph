import unittest
import torch
import torch.optim as optim
import torch.nn as nn
from unittest.mock import Mock, patch, call, MagicMock, ANY

# --- Original Source Code (made self-contained for the test file) ---


# Simulated src/shared/interfaces.py
class ILogger:
    """A simple interface for logging, simulated for testing."""

    def info(self, message: str):
        """Logs an informational message."""
        pass  # In a real implementation, this would log.

    def error(self, message: str):
        """Logs an error message."""
        pass  # In a real implementation, this would log.


# Simulated src/backend/models/graph_models.py (Minimal GraphSAGEModel)
class GraphSAGEModel(nn.Module):
    """
    A minimal implementation of GraphSAGEModel for testing purposes.
    It provides dummy forward and link prediction methods to satisfy the trainer's needs.
    """

    def __init__(self, input_dim=32, hidden_dim=64, output_dim=64):
        super().__init__()
        # These are dummy layers to ensure the model has parameters
        # that the optimizer can track.
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Mocks the forward pass of GraphSAGE, returning dummy node embeddings.
        The shape ensures compatibility with subsequent link prediction.
        """
        # Returns a dummy tensor with a shape suitable for further processing
        return torch.randn(x.shape[0], self.lin2.out_features)

    def predict_link(
        self, z: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Mocks the link prediction, returning dummy logits for binary classification.
        The output shape matches what BCEWithLogitsLoss expects.
        """
        # Returns dummy logits for each potential link
        return torch.randn(edge_label_index.shape[1])


# Simulated src/backend/services/graph_training_service.py (Original LinkPredictionTrainer)
class LinkPredictionTrainer:
    """
    Trains a GraphSAGE model for link prediction.
    Handles negative sampling, device placement, and the optimization loop.
    """

    def __init__(self, model: GraphSAGEModel, data, config, logger: ILogger):
        self.model = model
        self.data = (
            data  # 'data' is expected to be a torch_geometric.data.Data-like object
        )
        self.config = config  # Configuration object with learning_rate and epochs
        self.logger = logger  # Logger for status updates
        # Initialize Adam optimizer with model parameters and configured learning rate
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        # Initialize Binary Cross-Entropy with Logits Loss for link prediction
        self.criterion = nn.BCEWithLogitsLoss()

    def _get_negative_samples(self) -> torch.Tensor:
        """
        Generates random negative edge samples.
        Negative samples are pairs of nodes that do not have an edge between them.
        """
        # Generates a tensor of shape (2, num_edges) with random node indices.
        # This simulates drawing non-existent edges.
        return torch.randint(
            0, self.data.num_nodes, (2, self.data.num_edges), dtype=torch.long
        )

    def train(self):
        """
        Executes the training loop for link prediction.
        Yields the current epoch number and the training loss for each epoch.
        """
        # Concatenate positive (existing) edges with generated negative samples
        # This creates the full set of edges for which to predict labels.
        edge_label_index = torch.cat(
            [self.data.edge_index, self._get_negative_samples()], dim=1
        )
        # Create corresponding labels: 1 for positive edges, 0 for negative edges
        edge_label = torch.cat(
            [torch.ones(self.data.num_edges), torch.zeros(self.data.num_edges)], dim=0
        )

        # Determine the device (GPU if available, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Training on device: {device}")

        # Move model and data to the determined device
        self.model.to(device)
        self.data = self.data.to(device)  # Requires data object to have a .to() method
        edge_label_index = edge_label_index.to(device)
        edge_label = edge_label.to(device)

        # Training loop
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()  # Set model to training mode
            self.optimizer.zero_grad()  # Clear gradients

            # Forward pass: get node embeddings
            z = self.model(self.data.x, self.data.edge_index)
            # Predict links using the node embeddings
            out = self.model.predict_link(z, edge_label_index)
            # Calculate the loss
            loss = self.criterion(out, edge_label)

            loss.backward()  # Backpropagate the loss
            self.optimizer.step()  # Update model parameters

            # Yield current epoch and loss value
            yield epoch, loss.item()


# --- Unit Test Code ---


class MockTorchGeometricData:
    """
    A mock class that simulates essential attributes and the `.to()` method
    of `torch_geometric.data.Data` for testing purposes.
    """

    def __init__(
        self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, num_edges: int
    ):
        self.x = x
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def to(self, device: str):
        """
        Simulates moving the data tensors (x, edge_index) to a specified device.
        This mock version just returns self to allow method chaining, without
        actually moving real tensors (which would require a GPU for 'cuda').
        """
        # In a real scenario, these would be deep copies to ensure no side effects
        # on the original tensors in the test setup.
        # For a mock, simply returning self is often sufficient.
        return self


class MockConfig:
    """A simple mock for the configuration object, providing learning rate and epochs."""

    def __init__(self, learning_rate: float = 0.01, epochs: int = 5):
        self.learning_rate = learning_rate
        self.epochs = epochs


class TestLinkPredictionTrainer(unittest.TestCase):
    """
    Unit tests for the LinkPredictionTrainer class.
    Uses unittest.mock for isolating the trainer's logic.
    """

    def setUp(self):
        """
        Set up common dependencies and instances for each test method.
        This method is called before every test.
        """
        # Initialize the simulated GraphSAGEModel
        self.model = GraphSAGEModel()
        # Mock model.parameters() to return a stable mock object
        self.mock_model_parameters = Mock(return_value=[Mock(spec=nn.Parameter)])
        self.model.parameters = self.mock_model_parameters

        # Initialize the mock data object resembling torch_geometric.data.Data
        self.data = MockTorchGeometricData(
            x=torch.randn(10, 32),  # 10 nodes, 32 features
            edge_index=torch.randint(0, 10, (2, 5)),  # 5 edges
            num_nodes=10,
            num_edges=5,
        )
        # Initialize the mock configuration
        self.config = MockConfig()
        # Initialize a mock for the ILogger interface
        self.logger = Mock(spec=ILogger)

        # Create the trainer instance, which will be tested
        # Note: For tests patching Adam/BCEWithLogitsLoss, we will re-instantiate trainer
        # inside the test method to ensure the patches are active during __init__.
        self.trainer = LinkPredictionTrainer(
            self.model, self.data, self.config, self.logger
        )

    @patch("torch.optim.Adam")
    @patch("torch.nn.BCEWithLogitsLoss")
    def test_link_prediction_trainer_init(self, MockBCEWithLogitsLoss, MockAdam):
        """
        Tests that the LinkPredictionTrainer initializes its attributes correctly,
        including the optimizer and loss criterion.
        """
        # Re-create trainer to ensure the patches are applied during init
        trainer = LinkPredictionTrainer(self.model, self.data, self.config, self.logger)

        self.assertIs(trainer.model, self.model)
        self.assertIs(trainer.data, self.data)
        self.assertIs(trainer.config, self.config)
        self.assertIs(trainer.logger, self.logger)

        # Assert that Adam and BCEWithLogitsLoss were instantiated with correct arguments
        # Use self.mock_model_parameters for the assertion since model.parameters() is mocked
        MockAdam.assert_called_once_with(
            self.mock_model_parameters.return_value, lr=self.config.learning_rate
        )
        MockBCEWithLogitsLoss.assert_called_once()

        # Assert that the trainer's optimizer and criterion are the mock instances
        self.assertIs(trainer.optimizer, MockAdam.return_value)
        self.assertIs(trainer.criterion, MockBCEWithLogitsLoss.return_value)

    # Adjust the return_value of mock_randint to match self.data.num_edges
    @patch("torch.randint")
    def test_get_negative_samples(self, mock_randint):
        """
        Tests the `_get_negative_samples` method for correct output shape,
        data type, and value range.
        """
        # Set the return value of mock_randint dynamically based on data.num_edges
        mock_randint.return_value = torch.tensor(
            [[0] * self.data.num_edges, [1] * self.data.num_edges], dtype=torch.long
        )

        negative_samples = self.trainer._get_negative_samples()

        # Verify torch.randint was called with correct arguments
        mock_randint.assert_called_once_with(
            0, self.data.num_nodes, (2, self.data.num_edges), dtype=torch.long
        )

        # Verify the shape and data type of the returned tensor
        self.assertEqual(negative_samples.shape, (2, self.data.num_edges))
        self.assertEqual(negative_samples.dtype, torch.long)

        # Additional check for values based on the mock_randint's return_value
        self.assertTrue(torch.all(negative_samples >= 0))
        self.assertTrue(torch.all(negative_samples < self.data.num_nodes))

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.ones")  # Removed return_value, will be mocked for .to() later
    @patch("torch.zeros")  # Removed return_value, will be mocked for .to() later
    @patch("torch.cat")
    def test_train_loop_cpu(self, mock_cat, mock_zeros, mock_ones, mock_is_available):
        """
        Tests the main training loop on CPU, ensuring key methods are called
        correctly and values are yielded.
        """
        # Mock internal components of the trainer to control their behavior
        self.trainer.optimizer.zero_grad = Mock()
        self.trainer.optimizer.step = Mock()
        self.trainer.model.train = Mock()

        # Mock the `to` method on the model and data for device placement checks.
        # We don't wrap to avoid actual tensor movement which can cause errors.
        self.trainer.model.to = Mock(return_value=self.trainer.model)
        self.trainer.data.to = Mock(return_value=self.trainer.data)

        # Mock the outputs of model's forward and predict_link methods
        # Create a mock tensor that has a .backward() method
        mock_output_tensor = Mock(spec=torch.Tensor)
        mock_output_tensor.item.return_value = 0.5  # Dummy loss value
        mock_output_tensor.backward = Mock()

        self.trainer.model.forward = Mock(
            return_value=MagicMock(spec=torch.Tensor)
        )  # Return a mock tensor
        self.trainer.model.predict_link = Mock(
            return_value=MagicMock(spec=torch.Tensor)
        )  # Return a mock tensor

        # Mock the criterion call to return the mock_output_tensor
        self.trainer.criterion = Mock(return_value=mock_output_tensor)

        # Mock _get_negative_samples to return a mock tensor with .to()
        mock_neg_samples = Mock(spec=torch.Tensor)
        mock_neg_samples.shape = (2, self.data.num_edges)
        mock_neg_samples.dtype = torch.long
        mock_neg_samples.to.return_value = mock_neg_samples  # allow chaining .to()
        self.trainer._get_negative_samples = Mock(return_value=mock_neg_samples)

        # Setup mock_ones and mock_zeros to return tensors with a .to() method
        mock_ones.return_value = MagicMock(spec=torch.Tensor)
        mock_ones.return_value.to.return_value = mock_ones.return_value
        mock_zeros.return_value = MagicMock(spec=torch.Tensor)
        mock_zeros.return_value.to.return_value = mock_zeros.return_value

        # Setup mock_cat's behavior for edge_label_index and edge_label
        mock_edge_label_index = MagicMock(spec=torch.Tensor)
        mock_edge_label_index.to.return_value = (
            mock_edge_label_index  # Enable chaining .to()
        )
        mock_edge_label = MagicMock(spec=torch.Tensor)
        mock_edge_label.to.return_value = mock_edge_label  # Enable chaining .to()

        # Configure mock_cat's side_effect for its two calls
        mock_cat.side_effect = [
            mock_edge_label_index,  # First call for edge_label_index
            mock_edge_label,  # Second call for edge_label
        ]

        epochs_ran = 0
        # Iterate through the generator yielded by the train method
        for epoch, loss_item in self.trainer.train():
            epochs_ran += 1
            # Assert types and sequence of yielded values
            self.assertIsInstance(epoch, int)
            self.assertIsInstance(loss_item, float)
            self.assertEqual(epoch, epochs_ran)

            # Assert that key methods were called exactly once per epoch
            self.trainer.model.train.assert_called_once()
            self.trainer.optimizer.zero_grad.assert_called_once()
            self.trainer.model.forward.assert_called_once_with(
                self.data.x, self.data.edge_index
            )
            self.trainer.model.predict_link.assert_called_once()
            self.trainer.criterion.assert_called_once()  # Verify criterion was called
            mock_output_tensor.backward.assert_called_once()  # Verify loss.backward() was called
            self.trainer.optimizer.step.assert_called_once()

            # Reset mocks for the next iteration (if any)
            self.trainer.model.train.reset_mock()
            self.trainer.optimizer.zero_grad.reset_mock()
            self.trainer.model.forward.reset_mock()
            self.trainer.model.predict_link.reset_mock()
            self.trainer.criterion.reset_mock()
            mock_output_tensor.backward.reset_mock()
            self.trainer.optimizer.step.reset_mock()
            self.trainer._get_negative_samples.reset_mock()  # Reset _get_negative_samples too
            mock_cat.reset_mock()  # Reset mock_cat for next iteration

        # Verify that the correct number of epochs were run
        self.assertEqual(epochs_ran, self.config.epochs)

        # Assert that the `.to()` method was called once on the model and data
        # at the beginning of the training process, with "cpu" device.
        self.trainer.model.to.assert_called_once_with("cpu")
        self.trainer.data.to.assert_called_once_with("cpu")
        self.logger.info.assert_called_with("Training on device: cpu")

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.ones")
    @patch("torch.zeros")
    @patch("torch.cat")
    def test_train_loop_cuda(self, mock_cat, mock_zeros, mock_ones, mock_is_available):
        """
        Tests the training loop's device placement when CUDA is reported as available.
        Ensures `to("cuda")` is called.
        """
        # Mock internal components for the trainer
        self.trainer.optimizer.zero_grad = Mock()
        self.trainer.optimizer.step = Mock()
        self.trainer.model.train = Mock()

        # Mock the `to` method on the model and data to check device argument
        self.trainer.model.to = Mock(return_value=self.trainer.model)
        self.trainer.data.to = Mock(return_value=self.trainer.data)

        mock_output_tensor = Mock(spec=torch.Tensor)
        mock_output_tensor.item.return_value = 0.5
        mock_output_tensor.backward = Mock()
        self.trainer.model.forward = Mock(return_value=MagicMock(spec=torch.Tensor))
        self.trainer.model.predict_link = Mock(
            return_value=MagicMock(spec=torch.Tensor)
        )
        self.trainer.criterion = Mock(return_value=mock_output_tensor)

        mock_neg_samples = Mock(spec=torch.Tensor)
        mock_neg_samples.shape = (2, self.data.num_edges)
        mock_neg_samples.dtype = torch.long
        mock_neg_samples.to.return_value = mock_neg_samples
        self.trainer._get_negative_samples = Mock(return_value=mock_neg_samples)

        mock_ones.return_value = MagicMock(spec=torch.Tensor)
        mock_ones.return_value.to.return_value = mock_ones.return_value
        mock_zeros.return_value = MagicMock(spec=torch.Tensor)
        mock_zeros.return_value.to.return_value = mock_zeros.return_value

        mock_edge_label_index = MagicMock(spec=torch.Tensor)
        mock_edge_label_index.to.return_value = mock_edge_label_index
        mock_edge_label = MagicMock(spec=torch.Tensor)
        mock_edge_label.to.return_value = mock_edge_label

        mock_cat.side_effect = [mock_edge_label_index, mock_edge_label]

        # Run just one epoch to verify device placement.
        epochs_ran = 0
        for epoch, loss_item in self.trainer.train():
            epochs_ran += 1
            break  # Exit after the first epoch for efficiency

        # Assert that the `.to()` method was called with "cuda" on the model and data
        self.trainer.model.to.assert_called_once_with("cuda")
        self.trainer.data.to.assert_called_once_with("cuda")
        self.logger.info.assert_called_with("Training on device: cuda")
