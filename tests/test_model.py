from unittest.mock import MagicMock, patch

from ftml.model import load_model_and_tokenizer


class TestLoadModelAndTokenizer:
    @patch("ftml.model._load_transformers")
    def test_dispatches_to_transformers(self, mock_load: MagicMock) -> None:
        mock_load.return_value = (MagicMock(), MagicMock())

        load_model_and_tokenizer("test/model", "hf_token", use_unsloth=False)

        mock_load.assert_called_once_with("test/model", "hf_token", True, False)

    @patch("ftml.model._load_unsloth")
    def test_dispatches_to_unsloth(self, mock_load: MagicMock) -> None:
        mock_load.return_value = (MagicMock(), MagicMock())

        load_model_and_tokenizer("test/model", "hf_token", use_unsloth=True, max_seq_length=1024)

        mock_load.assert_called_once_with("test/model", True, 1024)

    @patch("ftml.model._load_transformers")
    def test_empty_token_passed_as_none(self, mock_load: MagicMock) -> None:
        mock_load.return_value = (MagicMock(), MagicMock())

        load_model_and_tokenizer("test/model", "", use_unsloth=False)

        mock_load.assert_called_once_with("test/model", None, True, False)

    @patch("ftml.model._load_transformers")
    def test_4bit_disabled(self, mock_load: MagicMock) -> None:
        mock_load.return_value = (MagicMock(), MagicMock())

        load_model_and_tokenizer("test/model", "tok", use_4bit=False, use_unsloth=False)

        mock_load.assert_called_once_with("test/model", "tok", False, False)

    @patch("ftml.model._load_transformers")
    def test_flash_attention_enabled(self, mock_load: MagicMock) -> None:
        mock_load.return_value = (MagicMock(), MagicMock())

        load_model_and_tokenizer(
            "test/model",
            "tok",
            use_unsloth=False,
            use_flash_attention=True,
        )

        mock_load.assert_called_once_with("test/model", "tok", True, True)
