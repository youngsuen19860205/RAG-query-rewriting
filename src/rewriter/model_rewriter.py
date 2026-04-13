"""
小模型改写器 — mT5-small 推理（支持 ONNX / INT8 量化）
优化目标: 推理延迟最低，适合端侧/在线场景
依赖: transformers, optimum[onnxruntime], onnxruntime
"""
import os
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# HuggingFace 模型标识（可本地路径覆盖）
DEFAULT_MODEL_ID = os.getenv("REWRITER_MODEL_ID", "google/mt5-small")
DEFAULT_ONNX_DIR = os.getenv("REWRITER_ONNX_DIR", "/tmp/mt5_small_onnx")
MAX_INPUT_LEN = 256
MAX_OUTPUT_LEN = 64


class ModelRewriter:
    """
    mT5-small 推理改写器

    优先加载 ONNX/INT8 量化版本以获得最低延迟；
    如 ONNX 不可用则回退到 PyTorch 推理。

    Note: When using the default model_id (google/mt5-small), the model weights
    will be downloaded from HuggingFace Hub on first run (~300MB). Set
    REWRITER_MODEL_ID env var or pass model_id to use a local path instead.

    用法:
        rewriter = ModelRewriter()
        rewriter.load()
        result, latency_ms = rewriter.rewrite(
            query="给他打电话",
            context_text="对话历史: 用户: 联系张三"
        )
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        onnx_dir: str = DEFAULT_ONNX_DIR,
        use_onnx: bool = True,
        use_int8: bool = True,
    ):
        self.model_id = model_id
        self.onnx_dir = onnx_dir
        self.use_onnx = use_onnx
        self.use_int8 = use_int8
        self._tokenizer = None
        self._model = None
        self._backend = None   # "onnx" | "torch"
        if model_id == DEFAULT_MODEL_ID:
            logger.info(
                "Using default model '%s'. On first run, weights (~300MB) will be "
                "downloaded from HuggingFace Hub. Set REWRITER_MODEL_ID env var to use a local path.",
                DEFAULT_MODEL_ID,
            )

    # ── 加载 ────────────────────────────────
    def load(self) -> None:
        """加载模型（优先 ONNX，回退 PyTorch）"""
        if self.use_onnx:
            loaded = self._load_onnx()
            if loaded:
                return
        self._load_torch()

    def _load_onnx(self) -> bool:
        """尝试加载 ONNX 量化模型"""
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            from transformers import AutoTokenizer

            onnx_path = self.onnx_dir
            if not os.path.isdir(onnx_path):
                logger.info("ONNX dir not found, exporting from %s ...", self.model_id)
                self._export_onnx()

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = ORTModelForSeq2SeqLM.from_pretrained(onnx_path)
            self._backend = "onnx"
            logger.info("ModelRewriter loaded: ONNX backend @ %s", onnx_path)
            return True
        except Exception as e:
            logger.warning("ONNX load failed: %s. Falling back to PyTorch.", e)
            return False

    def _load_torch(self) -> None:
        """加载 PyTorch 模型"""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self._model.eval()
        self._backend = "torch"
        logger.info("ModelRewriter loaded: PyTorch backend @ %s", self.model_id)

    def _export_onnx(self) -> None:
        """导出并量化为 ONNX INT8"""
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from optimum.onnxruntime import ORTQuantizer

        os.makedirs(self.onnx_dir, exist_ok=True)
        model = ORTModelForSeq2SeqLM.from_pretrained(
            self.model_id, export=True
        )
        model.save_pretrained(self.onnx_dir)

        if self.use_int8:
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
            for component in ["encoder_model", "decoder_model", "decoder_with_past_model"]:
                onnx_file = os.path.join(self.onnx_dir, f"{component}.onnx")
                if os.path.exists(onnx_file):
                    quantizer = ORTQuantizer.from_pretrained(self.onnx_dir, file_name=f"{component}.onnx")
                    quantizer.quantize(save_dir=self.onnx_dir, quantization_config=qconfig)

        logger.info("ONNX export complete: %s", self.onnx_dir)

    # ── 推理 ────────────────────────────────
    def _build_input(self, query: str, context_text: str = "") -> str:
        """构建模型输入 prompt"""
        if context_text:
            return f"改写query: {context_text} | 当前: {query}"
        return f"改写query: {query}"

    def rewrite(
        self,
        query: str,
        context_text: str = "",
        num_beams: int = 2,
    ) -> Tuple[str, float]:
        """
        执行改写推理

        Returns:
            (rewritten_query, latency_ms)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        input_text = self._build_input(query, context_text)
        t0 = time.perf_counter()

        if self._backend == "onnx":
            result = self._infer_onnx(input_text, num_beams)
        else:
            result = self._infer_torch(input_text, num_beams)

        latency_ms = (time.perf_counter() - t0) * 1000
        return result, latency_ms

    def _infer_onnx(self, input_text: str, num_beams: int) -> str:
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
        )
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_LEN,
            num_beams=num_beams,
            early_stopping=True,
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _infer_torch(self, input_text: str, num_beams: int) -> str:
        import torch
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
        )
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT_LEN,
                num_beams=num_beams,
                early_stopping=True,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def backend(self) -> Optional[str]:
        return self._backend
