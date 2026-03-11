from unittest.mock import MagicMock, patch

import torch

from ftml.model import build_quantization_config, load_model_and_tokenizer


class TestBuildQuantizationConfig:
    def test_returns_none_when_disabled(self) -> None:
        assert build_quantization_config(False) is None

    def test_4bit_config_params(self) -> None:
        config = build_quantization_config(True)
        assert config is not None
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True
        assert config.bnb_4bit_compute_dtype == torch.bfloat16


class TestLoadModelAndTokenizer:
    @patch("ftml.model.prepare_model_for_kbit_training")
    @patch("ftml.model.AutoTokenizer.from_pretrained")
    @patch("ftml.model.AutoModelForCausalLM.from_pretrained")
    def test_loads_model_and_tokenizer(
        self,
        mock_model_cls: MagicMock,
        mock_tok_cls: MagicMock,
        mock_prepare: MagicMock,
    ) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tok_cls.return_value = mock_tokenizer

        model, tokenizer = load_model_and_tokenizer("test/model", "hf_token123")

        mock_model_cls.assert_called_once()
        mock_tok_cls.assert_called_once_with("test/model", token="hf_token123")
        assert model is mock_model
        assert tokenizer is mock_tokenizer
        mock_prepare.assert_not_called()

    @patch("ftml.model.prepare_model_for_kbit_training")
    @patch("ftml.model.AutoTokenizer.from_pretrained")
    @patch("ftml.model.AutoModelForCausalLM.from_pretrained")
    def test_sets_pad_token_when_missing(
        self,
        mock_model_cls: MagicMock,
        mock_tok_cls: MagicMock,
        _mock_prepare: MagicMock,
    ) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2
        mock_tok_cls.return_value = mock_tokenizer

        loaded_model, tokenizer = load_model_and_tokenizer("test/model", "hf_token123")

        assert tokenizer.pad_token == "</s>"
        assert loaded_model.config.pad_token_id == 2

    @patch("ftml.model.prepare_model_for_kbit_training")
    @patch("ftml.model.AutoTokenizer.from_pretrained")
    @patch("ftml.model.AutoModelForCausalLM.from_pretrained")
    def test_prepares_kbit_training_when_quantized(
        self,
        mock_model_cls: MagicMock,
        mock_tok_cls: MagicMock,
        mock_prepare: MagicMock,
    ) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tok_cls.return_value = mock_tokenizer
        mock_prepare.return_value = mock_model

        quant_config = build_quantization_config(True)
        _model, _ = load_model_and_tokenizer("test/model", "", quant_config)

        mock_prepare.assert_called_once_with(mock_model)

    @patch("ftml.model.prepare_model_for_kbit_training")
    @patch("ftml.model.AutoTokenizer.from_pretrained")
    @patch("ftml.model.AutoModelForCausalLM.from_pretrained")
    def test_empty_token_passed_as_none(
        self,
        mock_model_cls: MagicMock,
        mock_tok_cls: MagicMock,
        _mock_prepare: MagicMock,
    ) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tok_cls.return_value = mock_tokenizer

        _model, _tokenizer = load_model_and_tokenizer("test/model", "")

        mock_tok_cls.assert_called_once_with("test/model", token=None)
