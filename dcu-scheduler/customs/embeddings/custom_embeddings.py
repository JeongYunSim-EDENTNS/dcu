from __future__ import annotations

import logging
import os
import warnings
from importlib.metadata import version
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_pydantic_field_names
from packaging.version import Version, parse
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _create_retry_decorator(embeddings: CustomEmbeddings) -> Callable[[Any], Any]:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _async_retry_decorator(embeddings: CustomEmbeddings) -> Any:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    async_retrying = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

    def wrap(func: Callable) -> Callable:
        async def wrapped_f(*args: Any, **kwargs: Any) -> Callable:
            async for _ in async_retrying:
                return await func(*args, **kwargs)
            raise AssertionError("this is unreachable")

        return wrapped_f

    return wrap


# https://stackoverflow.com/questions/76469415/getting-embeddings-of-length-1-from-langchain-openaiembeddings
def _check_response(response: dict, skip_empty: bool = False) -> dict:
    if any(len(d["embedding"]) == 1 for d in response["data"]) and not skip_empty:
        import openai

        raise openai.error.APIError("OpenAI API returned an empty embedding")
    return response


def embed_with_retry(embeddings: CustomEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    if _is_openai_v1():
        return embeddings.client.create(**kwargs)
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        response = embeddings.client.create(**kwargs)
        return _check_response(response, skip_empty=embeddings.skip_empty)

    return _embed_with_retry(**kwargs)


async def async_embed_with_retry(embeddings: CustomEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""

    if _is_openai_v1():
        return await embeddings.async_client.create(**kwargs)

    @_async_retry_decorator(embeddings)
    async def _async_embed_with_retry(**kwargs: Any) -> Any:
        response = await embeddings.client.acreate(**kwargs)
        return _check_response(response, skip_empty=embeddings.skip_empty)

    return await _async_embed_with_retry(**kwargs)


def _is_openai_v1() -> bool:
    _version = parse(version("openai"))
    return _version >= Version("1.0.0")


class CustomEmbeddings(BaseModel, Embeddings):
    """Custom embedding models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import CustomEmbeddings
            openai = CustomEmbeddings(openai_api_key="my-api-key")

    In order to use the library with Microsoft Azure endpoints, you need to set
    the OPENAI_API_TYPE, OPENAI_API_BASE, OPENAI_API_KEY and OPENAI_API_VERSION.
    The OPENAI_API_TYPE must be set to 'azure' and the others correspond to
    the properties of your endpoint.
    In addition, the deployment name must be passed as the model parameter.

    Example:
        .. code-block:: python

            import os

            os.environ["OPENAI_API_TYPE"] = "azure"
            os.environ["OPENAI_API_BASE"] = "https://<your-endpoint.openai.azure.com/"
            os.environ["OPENAI_API_KEY"] = "your AzureOpenAI key"
            os.environ["OPENAI_API_VERSION"] = "2023-05-15"
            os.environ["OPENAI_PROXY"] = "http://your-corporate-proxy:8080"

            from langchain.embeddings.openai import CustomEmbeddings
            embeddings = CustomEmbeddings(
                deployment="your-embeddings-deployment-name",
                model="your-embeddings-model-name",
                openai_api_base="https://your-endpoint.openai.azure.com/",
                openai_api_type="azure",
            )
            text = "This is a test query."
            query_result = embeddings.embed_query(text)

    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = "text-embedding-ada-002"
    # to support Azure OpenAI Service custom deployment names
    deployment: Optional[str] = model
    # TODO: Move to AzureCustomEmbeddings.
    openai_api_version: Optional[str] = Field(default=None, alias="api_version")
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided."""
    # to support Azure OpenAI Service custom endpoints
    openai_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    # to support Azure OpenAI Service custom endpoints
    openai_api_type: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    embedding_ctx_length: int = 8191
    """The maximum number of tokens to embed at once."""
    openai_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""
    openai_organization: Optional[str] = Field(default=None, alias="organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, httpx.Timeout or 
        None."""
    headers: Any = None
    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    skip_empty: bool = False
    """Whether to skip empty strings when embedding or raise an error.
    Defaults to not skipping."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    """Optional httpx.Client."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["openai_api_key"] = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        values["openai_api_base"] = values["openai_api_base"] or os.getenv(
            "OPENAI_API_BASE"
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values,
            "openai_api_type",
            "OPENAI_API_TYPE",
            default="",
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        if values["openai_api_type"] in ("azure", "azure_ad", "azuread"):
            default_api_version = "2023-05-15"
            # Azure Custom embedding models allow a maximum of 16 texts
            # at a time in each batch
            # See: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings
            values["chunk_size"] = min(values["chunk_size"], 16)
        else:
            default_api_version = ""
        values["openai_api_version"] = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
            default=default_api_version,
        )
        # Check OPENAI_ORGANIZATION for backwards compatibility.
        values["openai_organization"] = (
            values["openai_organization"]
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        else:
            if _is_openai_v1():
                if values["openai_api_type"] in ("azure", "azure_ad", "azuread"):
                    warnings.warn(
                        "If you have openai>=1.0.0 installed and are using Azure, "
                        "please use the `AzureCustomEmbeddings` class."
                    )
                client_params = {
                    "api_key": values["openai_api_key"],
                    "organization": values["openai_organization"],
                    "base_url": values["openai_api_base"],
                    "timeout": values["request_timeout"],
                    "max_retries": values["max_retries"],
                    "default_headers": values["default_headers"],
                    "default_query": values["default_query"],
                    "http_client": values["http_client"],
                }
                if not values.get("client"):
                    values["client"] = openai.OpenAI(**client_params).embeddings
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).embeddings
            elif not values.get("client"):
                values["client"] = openai.Embedding
            else:
                pass
        return values

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        if _is_openai_v1():
            openai_args: Dict = {"model": self.model, **self.model_kwargs}
        else:
            openai_args = {
                "model": self.model,
                "request_timeout": self.request_timeout,
                "headers": self.headers,
                "api_key": self.openai_api_key,
                "organization": self.openai_organization,
                "api_base": self.openai_api_base,
                "api_type": self.openai_api_type,
                "api_version": self.openai_api_version,
                **self.model_kwargs,
            }
            if self.openai_api_type in ("azure", "azure_ad", "azuread"):
                openai_args["engine"] = self.deployment
            # TODO: Look into proxy with openai v1.
            if self.openai_proxy:
                try:
                    import openai
                except ImportError:
                    raise ImportError(
                        "Could not import openai python package. "
                        "Please install it with `pip install openai`."
                    )

                openai.proxy = {
                    "http": self.openai_proxy,
                    "https": self.openai_proxy,
                }  # type: ignore[assignment]  # noqa: E501
        return openai_args

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None, headers: Optional[Dict[str, str]] = None
    ) -> List[List[float]]:
        batched_embeddings: List[List[float]] = []
        
        params = {
            "extra_headers": headers,
            **self._invocation_params
        }

        for text in texts:
            response = embed_with_retry(
                self,
                input=text,
                **params,
            )
            if not isinstance(response, dict):
                response = response.dict()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        return batched_embeddings

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    async def _aget_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None, headers: Optional[Dict[str, str]] = None
    ) -> List[List[float]]:
        batched_embeddings: List[List[float]] = []
        
        params = {
            "extra_headers": headers,
            **self._invocation_params
        }
        
        for text in texts:
            response = await async_embed_with_retry(
                self,
                input=text,
                **params,
            )
            if not isinstance(response, dict):
                response = response.dict()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        return batched_embeddings

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0, headers: Optional[Dict[str, str]] = None
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        engine = cast(str, self.deployment)
        return self._get_len_safe_embeddings(texts, engine=engine, headers=headers)

    async def aembed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0, headers: Optional[Dict[str, str]] = None
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint async for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        engine = cast(str, self.deployment)
        return await self._aget_len_safe_embeddings(texts, engine=engine, headers=headers)

    def embed_query(self, text: str, headers: Optional[Dict[str, str]] = None) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text], headers=headers)[0]

    async def aembed_query(self, text: str, headers: Optional[Dict[str, str]] = None) -> List[float]:
        """Call out to OpenAI's embedding endpoint async for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embeddings = await self.aembed_documents([text], headers=headers)
        return embeddings[0]
