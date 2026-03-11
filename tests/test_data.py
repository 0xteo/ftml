from unittest.mock import MagicMock, patch

from datasets import Dataset, DatasetDict

from ftml.data import format_for_sft, load_dataset_from_hf


class TestLoadDatasetFromHf:
    @patch("ftml.data._load_dataset")
    def test_creates_validation_split_when_missing(self, mock_load: MagicMock) -> None:
        ds = Dataset.from_dict({"text": [f"sample {i}" for i in range(100)]})
        mock_load.return_value = DatasetDict({"train": ds})

        result = load_dataset_from_hf("test/dataset", "hf_token")

        assert "train" in result
        assert "validation" in result
        assert len(result["train"]) + len(result["validation"]) == 100

    @patch("ftml.data._load_dataset")
    def test_preserves_existing_validation_split(self, mock_load: MagicMock) -> None:
        train_ds = Dataset.from_dict({"text": ["train sample"]})
        val_ds = Dataset.from_dict({"text": ["val sample"]})
        mock_load.return_value = DatasetDict({"train": train_ds, "validation": val_ds})

        result = load_dataset_from_hf("test/dataset", "hf_token")

        assert len(result["train"]) == 1
        assert len(result["validation"]) == 1

    @patch("ftml.data._load_dataset")
    def test_single_dataset_gets_split(self, mock_load: MagicMock) -> None:
        ds = Dataset.from_dict({"text": [f"sample {i}" for i in range(100)]})
        mock_load.return_value = ds

        result = load_dataset_from_hf("test/dataset", "")

        assert "train" in result
        assert "validation" in result


class TestFormatForSft:
    def test_conversations_format(self, mock_tokenizer: MagicMock) -> None:
        ds = Dataset.from_dict(
            {
                "conversations": [
                    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
                ],
            },
        )

        result = format_for_sft(ds, mock_tokenizer)

        assert "text" in result.column_names
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_messages_format(self, mock_tokenizer: MagicMock) -> None:
        ds = Dataset.from_dict(
            {
                "messages": [
                    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
                ],
            },
        )

        result = format_for_sft(ds, mock_tokenizer)

        assert "text" in result.column_names

    def test_text_format_passthrough(self, mock_tokenizer: MagicMock) -> None:
        ds = Dataset.from_dict({"text": ["Hello world"]})

        result = format_for_sft(ds, mock_tokenizer)

        assert result["text"] == ["Hello world"]

    def test_instruction_format(self, mock_tokenizer: MagicMock) -> None:
        ds = Dataset.from_dict(
            {
                "instruction": ["Translate to Bulgarian"],
                "output": ["Преведи на български"],
            },
        )

        result = format_for_sft(ds, mock_tokenizer)

        assert "text" in result.column_names
        assert "### Instruction:" in result["text"][0]
        assert "### Response:" in result["text"][0]

    def test_instruction_with_input(self, mock_tokenizer: MagicMock) -> None:
        ds = Dataset.from_dict(
            {
                "instruction": ["Translate"],
                "input": ["Hello"],
                "output": ["Здравей"],
            },
        )

        result = format_for_sft(ds, mock_tokenizer)

        assert "### Input:" in result["text"][0]

    def test_unsupported_format_raises(self, mock_tokenizer: MagicMock) -> None:
        ds = Dataset.from_dict({"unknown_col": ["data"]})

        import pytest

        with pytest.raises(ValueError, match="Unsupported dataset format"):
            format_for_sft(ds, mock_tokenizer)
