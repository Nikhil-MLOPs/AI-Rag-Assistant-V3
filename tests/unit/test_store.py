import pytest
from unittest.mock import MagicMock, patch
from src.embeddings.store import store_embeddings

# We patch Path.exists globally within the store module
@patch("src.embeddings.store.Path.exists") 
@patch("src.embeddings.store.get_vector_store")
@patch("src.embeddings.store.np.load")
@patch("builtins.open")
@patch("json.load")
def test_store_embeddings_batching(mock_json_load, mock_open, mock_np_load, mock_get_client, mock_exists):
    """Verifies that large datasets are split into chunks of 5000."""
    # Force the file check to pass
    mock_exists.return_value = True

    # Setup mock data (6000 items to trigger 2 batches)
    num_records = 6000
    mock_np_load.return_value = MagicMock(tolist=lambda: [[0.1]*384] * num_records)
    mock_json_load.return_value = [{"meta": "data"}] * num_records
    
    # Mock reading the chunks file
    mock_file = mock_open.return_value.__enter__.return_value
    mock_file.__iter__.return_value = ['{"text": "test content"}'] * num_records
    
    # Setup mock Chroma client
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_or_create_collection.return_value = mock_collection

    # Run the function
    store_embeddings()

    # Assertions
    assert mock_collection.upsert.call_count == 2
    
    # Verify the first batch size is exactly 5000
    args, kwargs = mock_collection.upsert.call_args_list[0]
    assert len(kwargs['ids']) == 5000