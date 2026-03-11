"""Tests for agent tools (HF Hub and hardware estimation)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestSearchModels:
    def test_returns_formatted(self):
        from ftml.agent.tools.hf_hub import search_models

        mock_model = MagicMock()
        mock_model.id = "org/model-7b"
        mock_model.downloads = 50000
        mock_model.likes = 100
        mock_model.tags = ["text-generation", "pytorch"]
        mock_model.safetensors = MagicMock()
        mock_model.safetensors.total = 7_000_000_000

        with patch("huggingface_hub.list_models", return_value=[mock_model]):
            result = search_models("test", task="text-generation", limit=5)

        assert "org/model-7b" in result
        assert "50,000" in result
        assert "7.0B" in result

    def test_empty_results(self):
        from ftml.agent.tools.hf_hub import search_models

        with patch("huggingface_hub.list_models", return_value=[]):
            result = search_models("nonexistent", task="text-generation", limit=5)

        assert "No models found" in result


class TestGetModelInfo:
    def test_all_fields(self):
        from ftml.agent.tools.hf_hub import get_model_info

        mock_info = MagicMock()
        mock_info.id = "org/model-7b"
        mock_info.safetensors = MagicMock()
        mock_info.safetensors.total = 7_000_000_000
        mock_info.pipeline_tag = "text-generation"
        mock_info.library_name = "transformers"
        mock_info.downloads = 50000
        mock_info.likes = 100
        mock_info.last_modified = "2025-01-01"
        mock_info.tags = ["pytorch", "text-generation"]
        mock_info.card_data = MagicMock()
        mock_info.card_data.language = "bg"
        mock_info.card_data.license = "apache-2.0"
        mock_info.card_data.base_model = "meta-llama/Llama-3-8B"

        with patch("huggingface_hub.model_info", return_value=mock_info):
            result = get_model_info("org/model-7b")

        assert "7.00B" in result
        assert "text-generation" in result
        assert "50,000" in result
        assert "bg" in result

    def test_missing_card(self):
        from ftml.agent.tools.hf_hub import get_model_info

        mock_info = MagicMock()
        mock_info.id = "org/model-7b"
        mock_info.safetensors = MagicMock()
        mock_info.safetensors.total = 7_000_000_000
        mock_info.pipeline_tag = "text-generation"
        mock_info.library_name = "transformers"
        mock_info.downloads = 1000
        mock_info.likes = 5
        mock_info.last_modified = "2025-01-01"
        mock_info.tags = []
        mock_info.card_data = None

        with patch("huggingface_hub.model_info", return_value=mock_info):
            result = get_model_info("org/model-7b")

        assert "org/model-7b" in result
        assert "Language" not in result


class TestSearchDatasets:
    def test_returns_formatted(self):
        from ftml.agent.tools.hf_hub import search_datasets

        mock_ds = MagicMock()
        mock_ds.id = "org/dataset-bg"
        mock_ds.downloads = 2000
        mock_ds.likes = 30
        mock_ds.tags = ["bg", "instruction"]

        with patch("huggingface_hub.list_datasets", return_value=[mock_ds]):
            result = search_datasets("Bulgarian", limit=5)

        assert "org/dataset-bg" in result
        assert "2,000" in result


class TestGetDatasetInfo:
    def test_all_fields(self):
        from ftml.agent.tools.hf_hub import get_dataset_info

        mock_info = MagicMock()
        mock_info.id = "org/dataset-bg"
        mock_info.downloads = 5000
        mock_info.likes = 50
        mock_info.last_modified = "2025-01-01"
        mock_info.tags = ["bg", "instruction", "chat"]
        mock_info.card_data = MagicMock()
        mock_info.card_data.language = "bg"
        mock_info.card_data.license = "cc-by-4.0"
        mock_info.card_data.task_categories = ["text-generation"]
        mock_info.card_data.size_categories = ["10K<n<100K"]

        with patch("huggingface_hub.dataset_info", return_value=mock_info):
            result = get_dataset_info("org/dataset-bg")

        assert "5,000" in result
        assert "bg" in result


class TestPreviewDataset:
    def test_shows_columns_and_content(self):
        from ftml.agent.tools.hf_hub import preview_dataset

        mock_rows = [
            {"text": "Hello world", "label": "greeting"},
            {"text": "Goodbye", "label": "farewell"},
        ]

        mock_ds = MagicMock()
        mock_ds.take.return_value = mock_rows

        with patch("datasets.load_dataset", return_value=mock_ds):
            result = preview_dataset("org/dataset", split="train", num_rows=2)

        assert "text" in result
        assert "label" in result
        assert "Hello world" in result
        assert "Row 1" in result


class TestEstimateVram:
    def test_7b_fits(self):
        from ftml.agent.tools.hardware import estimate_vram

        result = estimate_vram(
            num_params_billions=7.0,
            use_4bit=True,
            lora_r=16,
            max_seq_length=2048,
            batch_size=4,
            available_vram_gb=24.0,
        )

        assert "YES" in result

    def test_70b_no_fit(self):
        from ftml.agent.tools.hardware import estimate_vram

        result = estimate_vram(
            num_params_billions=70.0,
            use_4bit=True,
            lora_r=16,
            max_seq_length=2048,
            batch_size=4,
            available_vram_gb=24.0,
        )

        assert "NO" in result
