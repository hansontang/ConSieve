"""
LLM Client supporting vLLM server, local vLLM, and Qwen batch API modes.
"""

import asyncio
import json
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

from .api_audit import APIAuditLogger, audit_logger_from_config


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""


@dataclass
class BatchJobInfo:
    """Information about a batch job."""
    batch_id: str
    status: str  # "pending", "running", "completed", "failed"
    method_name: str
    total_requests: int
    completed_requests: int = 0
    created_at: str = ""
    output_file: str = ""
    error_message: str = ""


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any] = None,
    ) -> List[LLMResponse]:
        """Generate responses for a list of prompts."""
        pass

    @abstractmethod
    def generate_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_params: Dict[str, Any] = None,
    ) -> List[LLMResponse]:
        """Generate responses for a list of message conversations."""
        pass

    def get_tokenizer(self):
        """Get tokenizer if available."""
        return None


class VLLMServerClient(BaseLLMClient):
    """Client for vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = None,
        max_concurrency: int = 64,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self._tokenizer = None

    def generate(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any] = None,
    ) -> List[LLMResponse]:
        """Generate using completion API (for raw prompts)."""
        # Convert to chat format
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        return self.generate_chat(messages_list, sampling_params)

    def generate_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_params: Dict[str, Any] = None,
    ) -> List[LLMResponse]:
        """Generate using chat completion API."""
        return asyncio.run(
            self._generate_chat_async(messages_list, sampling_params or {})
        )

    async def _generate_chat_async(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_params: Dict[str, Any],
    ) -> List[LLMResponse]:
        """Async batch generation with concurrency control."""
        results = [None] * len(messages_list)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Filter sampling params for OpenAI compatibility
        params = {
            k: v for k, v in sampling_params.items()
            if k in {"max_tokens", "temperature", "top_p", "stop", "presence_penalty", "frequency_penalty"}
        }

        async def _call(idx: int, messages: List[Dict[str, str]]):
            async with semaphore:
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **params
                    )
                    choice = resp.choices[0]
                    results[idx] = LLMResponse(
                        text=choice.message.content.strip() if choice.message.content else "",
                        input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                        output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                        finish_reason=choice.finish_reason or "",
                    )
                except Exception as e:
                    error_text = str(e)
                    if "inappropriate content" in error_text.lower():
                        # Provider policy refusal is a reproducible output
                        # state, not an answer and not an infrastructure retry.
                        results[idx] = LLMResponse(
                            text="[CONTENT_FILTERED]",
                            finish_reason="content_filter",
                        )
                    else:
                        results[idx] = LLMResponse(
                            text=f"[ERROR: {error_text}]",
                            finish_reason="error",
                        )

        tasks = [_call(idx, msgs) for idx, msgs in enumerate(messages_list)]
        await asyncio.gather(*tasks)

        return results


class VLLMLocalClient(BaseLLMClient):
    """Client for local vLLM (Python API)."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm package required. Install with: pip install vllm")

        self.LLM = LLM
        self.SamplingParams = SamplingParams

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs,
        )
        self._tokenizer = None

    def get_tokenizer(self):
        """Get the tokenizer from vLLM."""
        if self._tokenizer is None:
            if hasattr(self.llm, "get_tokenizer"):
                self._tokenizer = self.llm.get_tokenizer()
        return self._tokenizer

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a prompt string using tokenizer's chat template."""
        tokenizer = self.get_tokenizer()
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        # Fallback to simple format
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def generate(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any] = None,
    ) -> List[LLMResponse]:
        """Generate for raw prompts."""
        params = sampling_params or {}
        sp = self.SamplingParams(
            max_tokens=params.get("max_tokens", 256),
            temperature=params.get("temperature", 0.0),
            top_p=params.get("top_p", 1.0),
            stop=params.get("stop"),
        )

        outputs = self.llm.generate(prompts, sp)

        results = []
        for output in outputs:
            text = output.outputs[0].text.strip() if output.outputs else ""

            # Get token counts
            input_tokens = len(output.prompt_token_ids) if hasattr(output, "prompt_token_ids") and output.prompt_token_ids else 0
            output_tokens = len(output.outputs[0].token_ids) if output.outputs and hasattr(output.outputs[0], "token_ids") else 0

            results.append(LLMResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=output.outputs[0].finish_reason if output.outputs else "",
            ))

        return results

    def generate_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_params: Dict[str, Any] = None,
    ) -> List[LLMResponse]:
        """Generate for chat messages."""
        prompts = [self._messages_to_prompt(msgs) for msgs in messages_list]
        return self.generate(prompts, sampling_params)


class QwenAPIClient(BaseLLMClient):
    """
    Client for Qwen/DashScope real-time API using OpenAI-compatible interface.

    This is the non-batch version that makes synchronous API calls with
    async concurrency for better throughput.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "qwen-turbo",
        base_url: str = None,
        max_concurrency: int = 32,
        request_timeout: float = 60.0,
        max_retries: int = 0,
        explicit_retries: int = 0,
        retry_backoff_seconds: float = 1.0,
        request_spacing_seconds: float = 0.0,
        audit_logger: Optional[APIAuditLogger] = None,
    ):
        """
        Initialize Qwen API client.

        Args:
            api_key: DashScope API key (or set DASHSCOPE_API_KEY env var)
            model_name: Model name (e.g., qwen-turbo, qwen-plus, qwen-max, qwen-flash)
            base_url: Optional base URL override
            max_concurrency: Maximum concurrent requests
            request_timeout: Per-request timeout in seconds
            max_retries: OpenAI client retry count; clean_v2 uses zero so retries are explicit
            explicit_retries: Audited retries after infrastructure failures
            retry_backoff_seconds: Base delay for explicit exponential backoff
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key required. Set DASHSCOPE_API_KEY env var or pass api_key."
            )

        self.model_name = model_name
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.max_concurrency = max_concurrency
        self.explicit_retries = max(0, int(explicit_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.request_spacing_seconds = max(0.0, float(request_spacing_seconds))
        self.audit_logger = audit_logger

        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=request_timeout,
                max_retries=max_retries,
            )
        except ImportError:
            raise ImportError(
                "openai package required for Qwen API. "
                "Install with: pip install openai"
            )

    def generate(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any] = None,
    ) -> List[LLMResponse]:
        """Generate using completion API (converts to chat format)."""
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        return self.generate_chat(messages_list, sampling_params)

    def generate_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_params: Dict[str, Any] = None,
    ) -> List[LLMResponse]:
        """Generate using chat completion API."""
        return asyncio.run(
            self._generate_chat_async(messages_list, sampling_params or {})
        )

    async def _generate_chat_async(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_params: Dict[str, Any],
    ) -> List[LLMResponse]:
        """Async batch generation with concurrency control."""
        results = [None] * len(messages_list)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Filter sampling params for OpenAI compatibility
        params = {
            k: v for k, v in sampling_params.items()
            if k in {"max_tokens", "temperature", "top_p", "stop", "presence_penalty", "frequency_penalty"}
        }

        async def _call(idx: int, messages: List[Dict[str, str]]):
            async with semaphore:
                request = {"model": self.model_name, "messages": messages, **params}
                for attempt in range(1, self.explicit_retries + 2):
                    request_id = self.audit_logger.next_request_id() if self.audit_logger else ""
                    started_at = time.perf_counter()
                    try:
                        resp = await self.client.chat.completions.create(**request)
                        choice = resp.choices[0]
                        results[idx] = LLMResponse(
                            text=choice.message.content.strip() if choice.message.content else "",
                            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                            finish_reason=choice.finish_reason or "",
                        )
                        if self.audit_logger:
                            self.audit_logger.log(
                                request_id=request_id,
                                request=request,
                                started_at=started_at,
                                status="success",
                                response=resp,
                                attempt=attempt,
                                metadata={"batch_index": idx, "explicit_retry": attempt > 1},
                            )
                        break
                    except Exception as e:
                        error_text = str(e)
                        is_content_filter = "inappropriate content" in error_text.lower()
                        is_final = is_content_filter or attempt > self.explicit_retries
                        if self.audit_logger:
                            self.audit_logger.log(
                                request_id=request_id,
                                request=request,
                                started_at=started_at,
                                status="error",
                                error=error_text,
                                attempt=attempt,
                                metadata={
                                    "batch_index": idx,
                                    "explicit_retry": attempt > 1,
                                    "will_retry": not is_final,
                                },
                            )
                        if is_final:
                            if is_content_filter:
                                results[idx] = LLMResponse(
                                    text="[CONTENT_FILTERED]",
                                    finish_reason="content_filter",
                                )
                            else:
                                results[idx] = LLMResponse(
                                    text=f"[ERROR: {error_text}]",
                                    finish_reason="error",
                                )
                            break
                        if self.retry_backoff_seconds:
                            await asyncio.sleep(
                                self.retry_backoff_seconds * (2 ** (attempt - 1))
                            )
                # Optional explicit pacing for providers with strict QPS
                # ramps. This is an execution control, not a hidden retry.
                if self.request_spacing_seconds:
                    await asyncio.sleep(self.request_spacing_seconds)

        tasks = [_call(idx, msgs) for idx, msgs in enumerate(messages_list)]
        await asyncio.gather(*tasks)

        return results


class QwenBatchClient:
    """
    Client for Qwen/DashScope batch API using OpenAI-compatible interface.

    This client supports asynchronous batch processing:
    1. Upload input file to get file_id
    2. Create batch job with file_id
    3. Poll for completion
    4. Download and parse results

    Batch API is typically 50% cheaper than real-time API.

    Test mode: Use model_name="batch-test-model" for testing.
    - Max 1MB file size, 100 rows per file, 2 concurrent tasks
    - Uses endpoint "/v1/chat/ds-test"
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "qwen-turbo",
        base_url: str = None,
        test_mode: bool = False,
    ):
        """
        Initialize Qwen batch client.

        Args:
            api_key: DashScope API key (or set DASHSCOPE_API_KEY env var)
            model_name: Model name (e.g., qwen-turbo, qwen-plus, qwen-max)
            base_url: Optional base URL override
            test_mode: If True, use batch-test-model for testing workflow
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key required. Set DASHSCOPE_API_KEY env var or pass api_key."
            )

        self.test_mode = test_mode
        if test_mode:
            self.model_name = "batch-test-model"
            self.endpoint = "/v1/chat/ds-test"
        else:
            self.model_name = model_name
            self.endpoint = "/v1/chat/completions"

        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # Use OpenAI SDK for API calls
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            raise ImportError(
                "openai package required for Qwen batch API. "
                "Install with: pip install openai"
            )

    def create_batch_input_file(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_params: Dict[str, Any] = None,
        custom_ids: List[str] = None,
    ) -> str:
        """
        Create a JSONL file for batch input.

        Args:
            messages_list: List of message conversations
            sampling_params: Sampling parameters
            custom_ids: Optional list of custom IDs for tracking

        Returns:
            Path to the created JSONL file
        """
        params = sampling_params or {}

        # Create temp file
        fd, filepath = tempfile.mkstemp(suffix=".jsonl", prefix="qwen_batch_")
        os.close(fd)

        with open(filepath, "w", encoding="utf-8") as f:
            for idx, messages in enumerate(messages_list):
                custom_id = custom_ids[idx] if custom_ids else f"request-{idx}"
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": self.endpoint,  # Use configured endpoint (test vs production)
                    "body": {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": params.get("max_tokens", 512),
                        "temperature": params.get("temperature", 0.0),
                        "top_p": params.get("top_p", 1.0),
                    }
                }
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        return filepath

    def submit_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_params: Dict[str, Any] = None,
        custom_ids: List[str] = None,
        description: str = None,
    ) -> BatchJobInfo:
        """
        Submit a batch job using OpenAI-compatible API.

        Args:
            messages_list: List of message conversations
            sampling_params: Sampling parameters
            custom_ids: Optional list of custom IDs for tracking
            description: Optional job description

        Returns:
            BatchJobInfo with batch_id for tracking
        """
        # Create input file
        input_file_path = self.create_batch_input_file(
            messages_list, sampling_params, custom_ids
        )

        try:
            # Step 1: Upload the input file
            with open(input_file_path, "rb") as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            input_file_id = file_response.id

            # Step 2: Create the batch job
            batch_response = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint=self.endpoint,
                completion_window="24h",
                metadata={
                    "description": description or f"Batch job for {len(messages_list)} requests"
                }
            )

            return BatchJobInfo(
                batch_id=batch_response.id,
                status="pending",
                method_name="",
                total_requests=len(messages_list),
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

        except Exception as e:
            raise RuntimeError(f"Batch submission failed: {str(e)}")

        finally:
            # Cleanup temp file
            try:
                os.unlink(input_file_path)
            except OSError:
                pass

    def check_batch_status(self, batch_id: str) -> BatchJobInfo:
        """
        Check the status of a batch job using OpenAI-compatible API.

        Args:
            batch_id: The batch job ID

        Returns:
            BatchJobInfo with current status
        """
        try:
            batch = self.client.batches.retrieve(batch_id)
        except Exception as e:
            return BatchJobInfo(
                batch_id=batch_id,
                status="failed",
                method_name="",
                total_requests=0,
                error_message=str(e),
            )

        # Map OpenAI batch status to our status
        status_map = {
            "validating": "pending",
            "in_progress": "running",
            "finalizing": "running",
            "completed": "completed",
            "failed": "failed",
            "expired": "failed",
            "cancelling": "failed",
            "cancelled": "failed",
        }

        # Get request counts
        request_counts = batch.request_counts
        total = request_counts.total if request_counts else 0
        completed = request_counts.completed if request_counts else 0

        # Get error message if any
        error_msg = ""
        if batch.errors and batch.errors.data:
            error_msg = "; ".join(
                e.message for e in batch.errors.data if hasattr(e, "message")
            )

        return BatchJobInfo(
            batch_id=batch_id,
            status=status_map.get(batch.status, batch.status),
            method_name="",
            total_requests=total,
            completed_requests=completed,
            output_file=batch.output_file_id or "",
            error_message=error_msg,
        )

    def retrieve_batch_results(
        self,
        batch_id: str,
        output_dir: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results from a completed batch job using OpenAI-compatible API.

        Args:
            batch_id: The batch job ID
            output_dir: Directory to save output file (optional)

        Returns:
            List of result dictionaries with custom_id and response
        """
        # Check status first
        status_info = self.check_batch_status(batch_id)
        if status_info.status != "completed":
            raise RuntimeError(
                f"Batch job not completed. Status: {status_info.status}, "
                f"Progress: {status_info.completed_requests}/{status_info.total_requests}"
            )

        output_file_id = status_info.output_file
        if not output_file_id:
            raise RuntimeError("No output file available for batch job")

        # Download results using file content API
        try:
            file_content = self.client.files.content(output_file_id)
            content_text = file_content.text
        except Exception as e:
            raise RuntimeError(f"Failed to download results: {str(e)}")

        # Optionally save to file
        if output_dir:
            output_path = Path(output_dir) / f"batch_{batch_id}_results.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content_text)

        # Parse results
        results = []
        for line in content_text.strip().split("\n"):
            if line.strip():
                results.append(json.loads(line))

        return results

    def parse_batch_results(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, LLMResponse]:
        """
        Parse batch results into LLMResponse objects.

        Args:
            results: Raw results from retrieve_batch_results

        Returns:
            Dict mapping custom_id to LLMResponse
        """
        parsed = {}

        for result in results:
            custom_id = result.get("custom_id", "")
            response_body = result.get("response", {}).get("body", {})
            error = result.get("error")

            if error:
                parsed[custom_id] = LLMResponse(
                    text=f"[ERROR: {error.get('message', 'Unknown error')}]",
                    finish_reason="error",
                )
            else:
                choices = response_body.get("choices", [])
                usage = response_body.get("usage", {})

                if choices:
                    choice = choices[0]
                    message = choice.get("message", {})
                    parsed[custom_id] = LLMResponse(
                        text=message.get("content", "").strip(),
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        finish_reason=choice.get("finish_reason", ""),
                    )
                else:
                    parsed[custom_id] = LLMResponse(
                        text="[ERROR: No response]",
                        finish_reason="error",
                    )

        return parsed

    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = 30,
        timeout: int = 3600,
        progress_callback=None,
    ) -> BatchJobInfo:
        """
        Wait for a batch job to complete.

        Args:
            batch_id: The batch job ID
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            progress_callback: Optional callback(completed, total)

        Returns:
            BatchJobInfo with final status
        """
        start_time = time.time()

        while True:
            status = self.check_batch_status(batch_id)

            if progress_callback:
                progress_callback(status.completed_requests, status.total_requests)

            if status.status in ("completed", "failed"):
                return status

            if time.time() - start_time > timeout:
                return BatchJobInfo(
                    batch_id=batch_id,
                    status="timeout",
                    method_name="",
                    total_requests=status.total_requests,
                    completed_requests=status.completed_requests,
                    error_message=f"Timeout after {timeout} seconds",
                )

            time.sleep(poll_interval)


def create_llm_client(config: Dict[str, Any]) -> BaseLLMClient:
    """
    Factory function to create LLM client based on config.

    Args:
        config: LLM configuration dict with 'mode', 'local', 'server' keys

    Returns:
        BaseLLMClient instance
    """
    mode = config.get("mode", "server")

    if mode == "server":
        server_cfg = config.get("server", {})
        return VLLMServerClient(
            base_url=server_cfg.get("base_url", "http://localhost:8000/v1"),
            api_key=server_cfg.get("api_key", "EMPTY"),
            model_name=server_cfg.get("model_name"),
            max_concurrency=server_cfg.get("max_concurrency", 64),
        )
    elif mode == "local":
        local_cfg = config.get("local", {})
        return VLLMLocalClient(
            model_name=local_cfg.get("model_name"),
            tensor_parallel_size=local_cfg.get("tensor_parallel_size", 1),
            max_model_len=local_cfg.get("max_model_len", 32768),
            gpu_memory_utilization=local_cfg.get("gpu_memory_utilization", 0.9),
        )
    elif mode == "qwen":
        qwen_cfg = config.get("qwen", {})
        audit_logger = audit_logger_from_config(config)
        return QwenAPIClient(
            api_key=qwen_cfg.get("api_key"),
            model_name=qwen_cfg.get("model_name", "qwen-turbo"),
            base_url=qwen_cfg.get("base_url"),
            max_concurrency=qwen_cfg.get("max_concurrency", 32),
            request_timeout=qwen_cfg.get("request_timeout", 60.0),
            max_retries=qwen_cfg.get("max_retries", 0),
            explicit_retries=qwen_cfg.get("explicit_retries", 0),
            retry_backoff_seconds=qwen_cfg.get("retry_backoff_seconds", 1.0),
            request_spacing_seconds=qwen_cfg.get("request_spacing_seconds", 0.0),
            audit_logger=audit_logger,
        )
    else:
        raise ValueError(f"Unknown LLM mode: {mode}")


class BatchGenerator:
    """
    Helper class for batched LLM generation with progress tracking.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        batch_size: int = 64,
        sampling_params: Dict[str, Any] = None,
    ):
        self.client = client
        self.batch_size = batch_size
        self.sampling_params = sampling_params or {}

    def generate_all(
        self,
        messages_list: List[List[Dict[str, str]]],
        desc: str = "Generating",
    ) -> List[LLMResponse]:
        """
        Generate responses for all messages with batching and progress bar.
        """
        results = []
        total = len(messages_list)

        for start in tqdm(range(0, total, self.batch_size), desc=desc):
            batch = messages_list[start:start + self.batch_size]
            batch_results = self.client.generate_chat(batch, self.sampling_params)
            results.extend(batch_results)

        return results
