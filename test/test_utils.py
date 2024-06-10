from src.utils import UtilsLLM
from unittest.mock import MagicMock, patch


class TestUtilsLLM:
    """ Unit test for UtilsLLM class functions """

    def test_check_if_pinecone_index_exists(self):
        utils = UtilsLLM()

        # Mock the PineconeClient and its list_indexes method
        pinecone_client_mock = MagicMock()
        pinecone_client_mock.list_indexes.return_value.names.return_value = [
            "index1",
            "index2",
        ]
        utils.list_indexes = pinecone_client_mock.list_indexes

        # Test case where the index exists
        result, message = utils.check_if_a_specific_index_exists_in_pinecone(
            pinecone_client_mock, "index1"
        )
        assert result is True
        assert message == "Pinecone index exists."

        # Test case where the index does not exist
        result, message = utils.check_if_a_specific_index_exists_in_pinecone(
            pinecone_client_mock, "non_existent_index"
        )
        assert result is False
        assert message == "Pinecone index not found."

    def test_create_pinecone_index(self):
        utils = UtilsLLM()

        # Mock the PineconeClient and its methods
        pinecone_client_mock = MagicMock()
        utils.check_if_a_specific_index_exists_in_pinecone = MagicMock(
            return_value=False
        )
        utils.create_index = MagicMock()

        # Patch the ServerlessSpec class
        with patch("pinecone.ServerlessSpec") as mock_serverless_spec:
            mock_serverless_spec.return_value = MagicMock()

            # Call the function
            result, message = utils.create_pinecone_index(
                pinecone_client_mock, "index1"
            )

            # Assert the function return value and message
            assert result is True
            assert message == "Succeed creating pinecone index"
