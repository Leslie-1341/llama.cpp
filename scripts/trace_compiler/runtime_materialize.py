#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT-trace-1B-A model-bound token transcript materializer.

Gate 1A deliberately emits placeholder block identifiers.  This module is the
single boundary that turns a *strictly validated* 1A v2 slice into a frozen
native-token transcript.  It never reconstructs text from Alibaba hash ids and
never falls back to text prompts: calibration uses ``/tokenize`` only, while
reference replay uses native ``/completion`` with an integer-token prompt.

A transcript produced with a fixture/fake server is intentionally marked
``UNVERIFIED``/``OPEN``.  Only an explicitly real calibration, a real
``/props`` + ``/slots`` direct-token preflight, and a complete Resident
reference pass may produce the model-bound state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from .common import BLOCK_SIZE_TOKENS


TRANSCRIPT_SCHEMA_VERSION = "gt-trace-1b-a/v1"
DIRECT_TOKEN_CONTRACT_VERSION = "llama-server-native-direct-token/v1"
MATERIALIZE_MODE_OPEN = "OPEN"
# This is direct integer-token materialization, not text tokenization.
MATERIALIZE_MODE_REAL = "direct_token_ids"
MATERIALIZATION_STATUS_UNVERIFIED = "UNVERIFIED"
MATERIALIZATION_STATUS_REAL = "MODEL_BOUND_REAL"
REFERENCE_PASS_NAME = "resident_reference"
MATERIALIZER_IDENTITY = "sha256-prf-safe-alphabet/v2"
# Reference completion is generated for model-bound qualification observation
# only.  It must NOT be used to construct a later turn's prompt: the Alibaba
# trace semantics (docs/qa-context-growth-pattern.md) prove that the next
# turn's input may strip special tokens from the previous completion, so
# ``previous.prompt + previous.reference_completion`` is not a stable prefix
# invariant of the next turn's input.
REFERENCE_COMPLETION_ROLE = "qualification_observation_only"
FORMAL_CORRECTNESS_ORACLE = "future_resident_multi_session_baseline"
RUNTIME_CONTRACT_KEYS = {
    "ctx_size",
    "executor",
    "kv_unified",
    "parallel",
    "cache_type_k",
    "cache_type_v",
    "no_cache_idle_slots",
    "no_context_shift",
    "paged_block_size",
    "action_target_bytes",
    "max_blocks",
}


def _strict_runtime_contract(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != RUNTIME_CONTRACT_KEYS:
        raise MaterializationError(f"{label} schema mismatch")
    contract = dict(value)
    if isinstance(contract["ctx_size"], bool) or not isinstance(contract["ctx_size"], int) or contract["ctx_size"] <= 0:
        raise MaterializationError(f"{label}.ctx_size is invalid")
    if not isinstance(contract["executor"], str) or not contract["executor"]:
        raise MaterializationError(f"{label}.executor is invalid")
    for key in ("kv_unified", "no_cache_idle_slots", "no_context_shift"):
        if contract[key] is not True:
            raise MaterializationError(f"{label}.{key} must be true")
    if contract["parallel"] != 3:
        raise MaterializationError(f"{label}.parallel must be 3")
    if contract["cache_type_k"] != "f16" or contract["cache_type_v"] != "f16":
        raise MaterializationError(f"{label} cache types must be f16")
    if contract["paged_block_size"] != 16:
        raise MaterializationError(f"{label}.paged_block_size must be 16")
    for key in ("action_target_bytes", "max_blocks"):
        if isinstance(contract[key], bool) or not isinstance(contract[key], int) or contract[key] <= 0:
            raise MaterializationError(f"{label}.{key} is invalid")
    return contract

# Fixed UTF-8 material.  It is intentionally small and stable: it is a
# calibration corpus, not a workload prompt and never enters formal replay.
DEFAULT_CALIBRATION_CORPUS: tuple[str, ...] = (
    "GT-trace-1B-A calibration: ASCII, numbers 0123456789.",
    "UTF-8 calibration: café naïve — Ελληνικά 中文 日本語.",
    "Whitespace\tand newline\ncalibration; punctuation !? [] {} ().",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_BLOCK_ID = re.compile(r"^L(-?[0-9]+):H(-?[0-9]+)$")


class MaterializationError(ValueError):
    """Base class for fail-closed transcript errors."""


class ParentArtifactError(MaterializationError):
    """The supplied GT-trace-1A parent is not a supported frozen artifact."""


class CalibrationError(MaterializationError):
    """Tokenizer calibration was missing, unstable, or unsafe."""


class DirectTokenContractError(MaterializationError):
    """The live server did not prove the native direct-token contract."""


class ServerProcessIdentityError(DirectTokenContractError):
    """The live endpoint cannot be bound to the requested server process."""


class ContextOverflowError(MaterializationError):
    """A request would exceed the server-authoritative context window."""


class PartialBlockConflictError(MaterializationError):
    """A carried partial block no longer matches observed frozen tokens."""


class CrossSessionIsolationError(MaterializationError):
    """Distinct sessions accidentally share a complete prompt prefix."""


class TranscriptDriftError(MaterializationError):
    """A policy prompt/completion differs from the frozen reference transcript."""


@dataclass(frozen=True)
class ApiResponse:
    """Small transport-neutral response envelope used by real and fake clients."""

    status_code: int
    payload: Any
    raw_body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any, status_code: int = 200) -> "ApiResponse":
        return cls(
            status_code=status_code,
            payload=payload,
            raw_body=canonical_json_bytes(payload),
        )


class ServerClient(Protocol):
    def get(self, path: str) -> ApiResponse | Mapping[str, Any]: ...

    def post(
        self, path: str, payload: Mapping[str, Any]
    ) -> ApiResponse | Mapping[str, Any]: ...


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically with explicit UTF-8 and no NaN values."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_response(value: ApiResponse | Mapping[str, Any]) -> ApiResponse:
    if isinstance(value, ApiResponse):
        return value
    if isinstance(value, Mapping):
        return ApiResponse.from_payload(dict(value))
    raise MaterializationError(
        f"server adapter returned unsupported response type {type(value).__name__}"
    )


def _require_mapping(response: ApiResponse, label: str) -> Mapping[str, Any]:
    if response.status_code != 200:
        raise MaterializationError(
            f"{label} failed with HTTP status {response.status_code}"
        )
    if not isinstance(response.payload, Mapping):
        raise MaterializationError(f"{label} response must be a JSON object")
    return response.payload


class LlamaServerClient:
    """Minimal native llama-server HTTP client.

    The client exposes only the endpoints needed by this gate.  It does not
    send text prompts to completion and does not implement an OpenAI fallback.
    """

    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> ApiResponse:
        url = self.base_url + (path if path.startswith("/") else "/" + path)
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = canonical_json_bytes(payload)
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as result:
                raw = result.read()
                status = int(result.status)
                response_headers = dict(result.headers.items())
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = int(exc.code)
            response_headers = dict(exc.headers.items()) if exc.headers else {}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise MaterializationError(f"HTTP {method} {path} failed: {exc}") from exc
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MaterializationError(
                f"HTTP {method} {path} returned non-JSON data"
            ) from exc
        return ApiResponse(status, parsed, raw, response_headers)

    def get(self, path: str) -> ApiResponse:
        return self._request("GET", path)

    def post(self, path: str, payload: Mapping[str, Any]) -> ApiResponse:
        return self._request("POST", path, payload)


@dataclass(frozen=True)
class FrozenTurn:
    lineage_id: int
    turn: int
    timestamp: float
    input_length: int
    request_plan: Mapping[str, int]
    type: str
    block_ids: tuple[str, ...]

    @property
    def n_predict(self) -> int:
        return int(self.request_plan["n_predict"])


@dataclass(frozen=True)
class FrozenSlice:
    """A strict, self-contained view of one GT-trace-1A v2 slice."""

    manifest: Mapping[str, Any]
    turns: tuple[FrozenTurn, ...]
    parent_manifest_sha256: str
    parent_event_stream_sha256: str
    manifest_path: str
    events_path: str

    @property
    def trace_repo_head(self) -> str:
        return str(self.manifest["trace_repo_head"])

    @property
    def trace_file(self) -> str:
        return str(self.manifest["trace_file"])

    @property
    def trace_file_sha256(self) -> str:
        return str(self.manifest["trace_file_sha256"])

    @property
    def session_ids(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.manifest["session_ids"])


def _strict_json_object(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParentArtifactError(f"{label} is not UTF-8") from exc

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ParentArtifactError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_pairs)
    except (json.JSONDecodeError, ParentArtifactError) as exc:
        if isinstance(exc, ParentArtifactError):
            raise
        raise ParentArtifactError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ParentArtifactError(f"{label} must contain a JSON object")
    return value


def _required_manifest_fields(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "slice_class",
        "trace_key",
        "trace_repo_head",
        "trace_file",
        "trace_file_sha256",
        "source_window_ts",
        "trace_span",
        "calibration_window",
        "evaluation_window",
        "selected_event_span",
        "selection_rule",
        "seed",
        "session_count",
        "turn_count",
        "time_dilation",
        "prefix_mode",
        "materialize_mode",
        "runtime_event_scope",
        "ttl_replay_scope",
        "tokenization_calibration_requirement",
        "ttl_calibration_identity",
        "max_prompt_tokens",
        "max_required_context_tokens",
        "max_context_footprint_tokens",
        "event_stream_sha256",
        "event_count",
        "session_ids",
        "descriptive_stats",
    }
    if manifest.get("schema_version") != "gt-trace-1a/v2":
        raise ParentArtifactError(
            "GT-trace-1B-A accepts only gt-trace-1a/v2 parent artifacts; "
            f"got {manifest.get('schema_version')!r}"
        )
    missing = sorted(required.difference(manifest))
    if missing:
        raise ParentArtifactError(f"1A manifest missing fields: {missing}")
    if manifest.get("prefix_mode") != "session_namespaced":
        raise ParentArtifactError("1A parent must use session_namespaced prefix mode")
    if manifest.get("materialize_mode") != "placeholder_token_ids":
        raise ParentArtifactError(
            "1A parent must remain placeholder_token_ids; materialized children "
            "must not be used as a new 1A parent"
        )
    if manifest.get("runtime_event_scope") != "arrival_and_request_plan_only":
        raise ParentArtifactError("1A runtime event scope mismatch")
    if manifest.get("ttl_replay_scope") != "offline_projected_characterization_only":
        raise ParentArtifactError("1A TTL replay scope mismatch")
    requirement = manifest.get("tokenization_calibration_requirement")
    if not isinstance(requirement, Mapping):
        raise ParentArtifactError("1A tokenization calibration requirement missing")
    if requirement.get("status") != "OPEN":
        raise ParentArtifactError("1A calibration requirement is not OPEN")


def _event_from_raw(raw: Mapping[str, Any], index: int) -> FrozenTurn:
    allowed = {
        "line_msg_index",
        "lineage_id",
        "lifecycle_generation",
        "event_type",
        "ts",
        "turn",
        "request_plan",
        "feature",
    }
    if set(raw) != allowed:
        raise ParentArtifactError(
            f"event {index} must have exactly {sorted(allowed)}, got {sorted(raw)}"
        )
    if raw.get("line_msg_index") != index:
        raise ParentArtifactError(f"event {index} line_msg_index is not contiguous")
    if raw.get("event_type") != "TURN_START":
        raise ParentArtifactError(
            "GT-trace-1B-A refuses projected/non-arrival events; "
            f"event {index} is {raw.get('event_type')!r}"
        )
    if raw.get("lifecycle_generation") != 1:
        raise ParentArtifactError("1A TURN_START lifecycle_generation must be 1")
    lineage_id = raw.get("lineage_id")
    turn = raw.get("turn")
    timestamp = raw.get("ts")
    if isinstance(lineage_id, bool) or not isinstance(lineage_id, int) or lineage_id < 0:
        raise ParentArtifactError(f"event {index} lineage_id is invalid")
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
        raise ParentArtifactError(f"event {index} turn is invalid")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise ParentArtifactError(f"event {index} timestamp is invalid")
    if timestamp < 0:
        raise ParentArtifactError(f"event {index} timestamp must be non-negative")
    feature = raw.get("feature")
    if not isinstance(feature, Mapping):
        raise ParentArtifactError(f"event {index} feature must be an object")
    feature_allowed = {
        "lineage_id",
        "turn",
        "relative_arrival_offset",
        "timestamp_abs",
        "input_length",
        "type",
        "block_count",
        "session_namespaced_block_ids",
    }
    if set(feature) != feature_allowed:
        raise ParentArtifactError(
            f"event {index} feature schema mismatch: got {sorted(feature)}"
        )
    if feature["lineage_id"] != lineage_id or feature["turn"] != turn:
        raise ParentArtifactError(f"event {index} feature identity disagrees with event")
    if feature["timestamp_abs"] != timestamp:
        raise ParentArtifactError(f"event {index} timestamp identity mismatch")
    input_length = feature.get("input_length")
    block_count = feature.get("block_count")
    if (
        isinstance(input_length, bool)
        or not isinstance(input_length, int)
        or input_length < 0
        or isinstance(block_count, bool)
        or not isinstance(block_count, int)
        or block_count < 0
    ):
        raise ParentArtifactError(f"event {index} input/block length is invalid")
    block_ids = feature.get("session_namespaced_block_ids")
    if not isinstance(block_ids, list) or len(block_ids) != block_count:
        raise ParentArtifactError(
            f"event {index} block materialization count mismatch: "
            f"expected {block_count}, got {len(block_ids) if isinstance(block_ids, list) else 'missing'}"
        )
    for block_id in block_ids:
        if not isinstance(block_id, str):
            raise ParentArtifactError(f"event {index} block id must be a string")
        match = _BLOCK_ID.fullmatch(block_id)
        if match is None or int(match.group(1)) != lineage_id:
            raise ParentArtifactError(
                f"event {index} block id is not namespaced to lineage {lineage_id}: {block_id!r}"
            )
    if input_length > block_count * BLOCK_SIZE_TOKENS:
        raise ParentArtifactError(
            f"event {index} input_length exceeds block constraint capacity"
        )
    if input_length > 0 and block_count == 0:
        raise ParentArtifactError(
            f"event {index} has positive input_length but no block constraint"
        )
    plan = raw.get("request_plan")
    if not isinstance(plan, Mapping):
        raise ParentArtifactError(f"event {index} request_plan is missing")
    if set(plan) != {"target_output_length", "n_predict", "required_context_tokens"}:
        raise ParentArtifactError(f"event {index} request_plan schema mismatch")
    values = [plan[name] for name in ("target_output_length", "n_predict", "required_context_tokens")]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ParentArtifactError(f"event {index} request_plan values are invalid")
    if plan["target_output_length"] != plan["n_predict"]:
        raise ParentArtifactError(f"event {index} n_predict differs from target output")
    if plan["required_context_tokens"] != input_length + plan["target_output_length"]:
        raise ParentArtifactError(f"event {index} required context is inconsistent")
    event_type = feature.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise ParentArtifactError(f"event {index} type is invalid")
    return FrozenTurn(
        lineage_id=lineage_id,
        turn=turn,
        timestamp=float(timestamp),
        input_length=input_length,
        request_plan={name: int(plan[name]) for name in plan},
        type=event_type,
        block_ids=tuple(block_ids),
    )


def load_frozen_slice(
    manifest_path: str | os.PathLike[str],
    events_path: Optional[str | os.PathLike[str]] = None,
) -> FrozenSlice:
    """Load and strictly validate one 1A v2 manifest/event pair.

    The loader hashes the supplied bytes and never consults the original
    Alibaba trace repository.  Legacy v1 artifacts are intentionally rejected.
    """
    manifest_file = Path(manifest_path)
    manifest_raw = manifest_file.read_bytes()
    manifest = _strict_json_object(manifest_raw, str(manifest_file))
    _required_manifest_fields(manifest)
    if events_path is None:
        mapping = {
            "typical": "events_typical.jsonl",
            "revisit-heavy": "events_revisit-heavy.jsonl",
            "burst-long-context": "events_burst-long-context.jsonl",
        }
        try:
            events_path = manifest_file.parent / mapping[str(manifest["slice_class"])]
        except KeyError as exc:
            raise ParentArtifactError("unknown 1A slice class") from exc
    events_file = Path(events_path)
    events_raw = events_file.read_bytes()
    expected_stream_sha = manifest.get("event_stream_sha256")
    actual_stream_sha = sha256_bytes(events_raw)
    if expected_stream_sha != actual_stream_sha:
        raise ParentArtifactError(
            "1A event stream SHA mismatch: "
            f"manifest={expected_stream_sha!r} actual={actual_stream_sha!r}"
        )
    turns: list[FrozenTurn] = []
    try:
        text = events_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParentArtifactError("1A event stream is not UTF-8") from exc
    for index, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ParentArtifactError(f"1A event stream contains a blank line at {index}")
        raw = _strict_json_object(line.encode("utf-8"), f"event line {index}")
        turns.append(_event_from_raw(raw, index))
    if len(turns) != manifest.get("event_count"):
        raise ParentArtifactError("1A event_count does not match event stream")
    if len(turns) != manifest.get("turn_count"):
        raise ParentArtifactError("1A turn_count does not match event stream")
    session_ids = manifest.get("session_ids")
    if not isinstance(session_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in session_ids
    ):
        raise ParentArtifactError("1A session_ids is invalid")
    if len(session_ids) != manifest.get("session_count"):
        raise ParentArtifactError("1A session_count does not match session_ids")
    if tuple(sorted(set(session_ids))) != tuple(session_ids):
        raise ParentArtifactError("1A session_ids must be sorted and unique")
    seen_sessions = {turn.lineage_id for turn in turns}
    if seen_sessions != set(session_ids):
        raise ParentArtifactError("1A session_ids do not match event lineages")
    by_session: dict[int, list[int]] = {}
    previous_ts: Optional[float] = None
    for turn in turns:
        if previous_ts is not None and turn.timestamp < previous_ts:
            raise ParentArtifactError("1A event timestamps are not ordered")
        previous_ts = turn.timestamp
        by_session.setdefault(turn.lineage_id, []).append(turn.turn)
    for lineage_id, session_turns in by_session.items():
        expected = list(range(1, len(session_turns) + 1))
        if session_turns != expected:
            raise ParentArtifactError(
                f"1A lineage {lineage_id} turn sequence is not contiguous"
            )
    if manifest.get("max_prompt_tokens", 0) < max(
        (turn.input_length for turn in turns), default=0
    ):
        raise ParentArtifactError("1A max_prompt_tokens is too small")
    if manifest.get("max_required_context_tokens", 0) < max(
        (turn.request_plan["required_context_tokens"] for turn in turns), default=0
    ):
        raise ParentArtifactError("1A max_required_context_tokens is too small")
    return FrozenSlice(
        manifest=dict(manifest),
        turns=tuple(turns),
        parent_manifest_sha256=sha256_bytes(manifest_raw),
        parent_event_stream_sha256=actual_stream_sha,
        manifest_path=str(manifest_file),
        events_path=str(events_file),
    )


@dataclass(frozen=True)
class TokenCalibration:
    corpus: tuple[str, ...]
    repetitions: int
    request_sha256: str
    response_sha256: str
    safe_alphabet: tuple[int, ...]
    safe_alphabet_sha256: str
    model_sha256: Optional[str]
    binary_sha256: Optional[str]
    real_model: bool
    special_token_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": list(self.corpus),
            "repetitions": self.repetitions,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "safe_alphabet": list(self.safe_alphabet),
            "safe_alphabet_sha256": self.safe_alphabet_sha256,
            "model_sha256": self.model_sha256,
            "binary_sha256": self.binary_sha256,
            "real_model": self.real_model,
            "special_token_ids": list(self.special_token_ids),
        }


def _extract_token_ids(payload: Mapping[str, Any], label: str) -> tuple[list[int], set[int]]:
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        raise CalibrationError(f"{label} response tokens must be a list")
    ids: list[int] = []
    special: set[int] = set()
    for item in tokens:
        if isinstance(item, bool):
            raise CalibrationError(f"{label} returned bool as token id")
        if isinstance(item, int):
            token_id = item
            is_special = False
        elif isinstance(item, Mapping):
            token_id = item.get("id")
            is_special = bool(item.get("special", False) or item.get("is_special", False))
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise CalibrationError(f"{label} returned an invalid token object")
        else:
            raise CalibrationError(f"{label} returned an invalid token entry")
        if token_id < 0:
            raise CalibrationError(f"{label} returned a negative token id")
        ids.append(token_id)
        if is_special:
            special.add(token_id)
    return ids, special


def _validate_identity_for_real(model_sha256: Optional[str], binary_sha256: Optional[str]) -> None:
    if not isinstance(model_sha256, str) or _HEX64.fullmatch(model_sha256) is None:
        raise CalibrationError("real calibration requires a 64-hex model SHA256")
    if not isinstance(binary_sha256, str) or _HEX64.fullmatch(binary_sha256) is None:
        raise CalibrationError("real calibration requires a 64-hex binary SHA256")


def _special_token_ids_from_props(payload: Mapping[str, Any]) -> set[int]:
    """Extract server-published special/EOG ids when the endpoint exposes them."""
    ids: set[int] = set()
    scalar_keys = (
        "bos_token",
        "eos_token",
        "bos_token_id",
        "eos_token_id",
        "eog_token",
        "eog_token_id",
    )
    list_keys = ("eog_tokens", "eog_token_ids", "eos_token_ids")
    for key in scalar_keys:
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            ids.add(value)
    for key in list_keys:
        values = payload.get(key)
        if isinstance(values, list):
            ids.update(
                value
                for value in values
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            )
    return ids


def calibrate_tokenizer(
    client: ServerClient,
    *,
    corpus: Sequence[str] = DEFAULT_CALIBRATION_CORPUS,
    repetitions: int = 2,
    special_token_ids: Iterable[int] = (),
    model_sha256: Optional[str] = None,
    binary_sha256: Optional[str] = None,
    real_model: bool = False,
) -> TokenCalibration:
    """Calibrate a target model's legal, non-special token alphabet.

    ``real_model`` is explicit by design.  A fake adapter can exercise the
    protocol but cannot accidentally produce a model-bound artifact.
    """
    if repetitions < 2:
        raise CalibrationError("token calibration requires at least two repetitions")
    corpus_tuple = tuple(str(text) for text in corpus)
    if not corpus_tuple or any(not text for text in corpus_tuple):
        raise CalibrationError("token calibration corpus must be non-empty")
    if real_model:
        if not isinstance(client, LlamaServerClient):
            raise CalibrationError(
                "only LlamaServerClient may mark calibration as real; "
                "fake adapters must remain UNVERIFIED"
            )
        _validate_identity_for_real(model_sha256, binary_sha256)
    known_special = {int(token_id) for token_id in special_token_ids}
    if hasattr(client, "get"):
        props_response = _as_response(client.get("/props"))
        if props_response.status_code == 200 and isinstance(props_response.payload, Mapping):
            known_special.update(_special_token_ids_from_props(props_response.payload))
    requests: list[dict[str, Any]] = []
    responses: list[Any] = []
    observed_ids: set[int] = set()
    observed_special: set[int] = set(known_special)
    for text in corpus_tuple:
        request = {
            "content": text,
            "add_special": False,
            "parse_special": False,
            "with_pieces": True,
        }
        baseline_ids: Optional[list[int]] = None
        for _ in range(repetitions):
            if hasattr(client, "tokenize"):
                result = getattr(client, "tokenize")(request)
            else:
                result = client.post("/tokenize", request)
            response = _as_response(result)
            payload = _require_mapping(response, "/tokenize")
            ids, response_special = _extract_token_ids(payload, "/tokenize")
            if baseline_ids is None:
                baseline_ids = ids
            elif ids != baseline_ids:
                raise CalibrationError(
                    "token calibration is unstable for fixed corpus text"
                )
            observed_special.update(response_special)
            requests.append(dict(request))
            responses.append(payload)
        assert baseline_ids is not None
        observed_ids.update(baseline_ids)
    safe_alphabet = tuple(sorted(observed_ids.difference(observed_special)))
    if not safe_alphabet:
        raise CalibrationError("calibration produced an empty safe token alphabet")
    if any(isinstance(token_id, bool) or token_id < 0 for token_id in safe_alphabet):
        raise CalibrationError("safe alphabet contains an illegal token id")
    return TokenCalibration(
        corpus=corpus_tuple,
        repetitions=repetitions,
        request_sha256=sha256_bytes(canonical_json_bytes(requests)),
        response_sha256=sha256_bytes(canonical_json_bytes(responses)),
        safe_alphabet=safe_alphabet,
        safe_alphabet_sha256=sha256_bytes(canonical_json_bytes(list(safe_alphabet))),
        model_sha256=model_sha256,
        binary_sha256=binary_sha256,
        real_model=real_model,
        special_token_ids=tuple(sorted(observed_special)),
    )


@dataclass(frozen=True)
class ServerAuthority:
    source: str
    slot_id: int
    props_n_ctx: int
    slot_n_ctx: int
    effective_n_ctx: int
    props_sha256: str
    slots_sha256: str
    probe_request_sha256: str
    probe_response_sha256: str
    contract_version: str = DIRECT_TOKEN_CONTRACT_VERSION
    props_model_path: Optional[str] = None
    process_identity: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "slot_id": self.slot_id,
            "props_n_ctx": self.props_n_ctx,
            "slot_n_ctx": self.slot_n_ctx,
            "effective_n_ctx": self.effective_n_ctx,
            "props_sha256": self.props_sha256,
            "slots_sha256": self.slots_sha256,
            "probe_request_sha256": self.probe_request_sha256,
            "probe_response_sha256": self.probe_response_sha256,
            "contract_version": self.contract_version,
            "props_model_path": self.props_model_path,
            "process_identity": self.process_identity,
        }

    def bind_process_identity(self, identity: dict[str, Any]) -> "ServerAuthority":
        return replace(self, process_identity=dict(identity))

    @classmethod
    def fixture(cls, effective_n_ctx: int, slot_id: int = 0) -> "ServerAuthority":
        if effective_n_ctx <= 0:
            raise DirectTokenContractError("fixture effective_n_ctx must be positive")
        return cls(
            source="fixture",
            slot_id=slot_id,
            props_n_ctx=effective_n_ctx,
            slot_n_ctx=effective_n_ctx,
            effective_n_ctx=effective_n_ctx,
            props_sha256="UNVERIFIED",
            slots_sha256="UNVERIFIED",
            probe_request_sha256="UNVERIFIED",
            probe_response_sha256="UNVERIFIED",
        )


def _n_ctx_from_props(props: Mapping[str, Any]) -> int:
    defaults = props.get("default_generation_settings")
    if not isinstance(defaults, Mapping):
        raise DirectTokenContractError("/props missing default_generation_settings")
    n_ctx = defaults.get("n_ctx")
    if n_ctx is None and isinstance(defaults.get("params"), Mapping):
        n_ctx = defaults["params"].get("n_ctx")
    if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx <= 0:
        raise DirectTokenContractError("/props has no positive authoritative n_ctx")
    return n_ctx


def preflight_direct_token_contract(
    client: ServerClient,
    *,
    probe_token_id: int,
    slot_id: int = 0,
) -> ServerAuthority:
    """Prove native integer-prompt completion and server context authority."""
    if isinstance(probe_token_id, bool) or not isinstance(probe_token_id, int) or probe_token_id < 0:
        raise DirectTokenContractError("probe token id is illegal")
    props_response = _as_response(client.get("/props"))
    props = _require_mapping(props_response, "/props")
    props_n_ctx = _n_ctx_from_props(props)
    slots_response = _as_response(client.get("/slots"))
    if slots_response.status_code != 200 or not isinstance(slots_response.payload, list):
        raise DirectTokenContractError("/slots did not return a JSON array")
    matching_slots = [
        slot
        for slot in slots_response.payload
        if isinstance(slot, Mapping) and slot.get("id") == slot_id
    ]
    if len(matching_slots) != 1:
        raise DirectTokenContractError(
            f"/slots did not expose exactly one requested slot id={slot_id}"
        )
    slot = matching_slots[0]
    slot_n_ctx = slot.get("n_ctx")
    if isinstance(slot_n_ctx, bool) or not isinstance(slot_n_ctx, int) or slot_n_ctx <= 0:
        raise DirectTokenContractError("/slots requested slot has no positive n_ctx")
    if slot.get("is_processing") is True:
        raise DirectTokenContractError("requested slot is already processing")
    if props_n_ctx != slot_n_ctx:
        raise DirectTokenContractError(
            f"/props n_ctx={props_n_ctx} disagrees with /slots n_ctx={slot_n_ctx}"
        )
    probe_request = {
        "prompt": [probe_token_id],
        # Do not use n_predict=0 as a prefill-only probe.  In the current
        # server revision the generation-budget stop check runs only after at
        # least one token has been decoded, so n_predict=0 can still produce
        # one token.  An explicit one-token generation probe has unambiguous
        # request/response semantics.
        "n_predict": 1,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 0,
        "cache_prompt": False,
        "return_tokens": True,
        "ignore_eos": True,
        "stream": False,
        "id_slot": slot_id,
    }
    probe_response = _as_response(client.post("/completion", probe_request))
    probe_payload = _require_mapping(probe_response, "/completion direct-token preflight")
    if probe_payload.get("id_slot") != slot_id:
        raise DirectTokenContractError(
            "direct-token preflight response id_slot does not match requested slot"
        )
    for field_name in ("tokens_evaluated", "tokens_predicted"):
        value = probe_payload.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DirectTokenContractError(
                f"direct-token preflight response missing valid {field_name}"
            )
    if probe_payload["tokens_evaluated"] != len(probe_request["prompt"]):
        raise DirectTokenContractError(
            "direct-token preflight did not consume the one-token prompt"
        )
    if probe_payload["tokens_predicted"] != 1:
        raise DirectTokenContractError(
            "direct-token preflight did not generate exactly one probe token"
        )
    generated = probe_payload.get("tokens")
    if not isinstance(generated, list) or len(generated) != 1:
        raise DirectTokenContractError(
            "direct-token preflight did not return exactly one generated token"
        )
    _validate_prompt_tokens(generated, "direct-token preflight completion")
    return ServerAuthority(
        source="server_props_and_slots",
        slot_id=slot_id,
        props_n_ctx=props_n_ctx,
        slot_n_ctx=slot_n_ctx,
        effective_n_ctx=slot_n_ctx,
        props_sha256=sha256_bytes(props_response.raw_body or canonical_json_bytes(props)),
        slots_sha256=sha256_bytes(
            slots_response.raw_body or canonical_json_bytes(slots_response.payload)
        ),
        probe_request_sha256=sha256_bytes(canonical_json_bytes(probe_request)),
        probe_response_sha256=sha256_bytes(
            probe_response.raw_body or canonical_json_bytes(probe_payload)
        ),
        props_model_path=(
            str(props["model_path"])
            if isinstance(props.get("model_path"), str)
            else None
        ),
    )


def _read_proc_stat_starttime(proc_dir: Path) -> str:
    """Read the Linux /proc stat starttime field as a stable process identity."""
    try:
        raw = (proc_dir / "stat").read_bytes()
    except OSError as exc:
        raise ServerProcessIdentityError(f"cannot read {proc_dir}/stat") from exc
    closing_paren = raw.rfind(b")")
    if closing_paren < 0:
        raise ServerProcessIdentityError(f"{proc_dir}/stat has no process name")
    fields = raw[closing_paren + 1 :].split()
    # The suffix starts at field 3 (state); field 22 is starttime.
    if len(fields) < 20:
        raise ServerProcessIdentityError(f"{proc_dir}/stat has no starttime field")
    starttime = fields[19].decode("ascii", errors="strict")
    if not starttime.isdigit():
        raise ServerProcessIdentityError(f"{proc_dir}/stat starttime is invalid")
    return starttime


def _read_proc_cmdline(proc_dir: Path) -> tuple[str, ...]:
    try:
        raw = (proc_dir / "cmdline").read_bytes()
    except OSError as exc:
        raise ServerProcessIdentityError(f"cannot read {proc_dir}/cmdline") from exc
    values = tuple(
        part.decode("utf-8", errors="surrogateescape")
        for part in raw.split(b"\0")
        if part
    )
    if not values:
        raise ServerProcessIdentityError(f"{proc_dir}/cmdline is empty")
    return values


def _resolve_process_arg(value: str, cwd: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = Path(cwd) / path
    return os.path.realpath(str(path))


def _cmdline_option_value(cmdline: Sequence[str], names: set[str]) -> Optional[str]:
    for index, arg in enumerate(cmdline):
        for name in names:
            prefix = name + "="
            if arg.startswith(prefix):
                return arg[len(prefix):]
        if arg in names and index + 1 < len(cmdline):
            return cmdline[index + 1]
    return None


def _cmdline_contains_path(cmdline: Sequence[str], expected: Path, cwd: str) -> bool:
    expected_real = os.path.realpath(str(expected))
    for arg in cmdline:
        candidates = [arg]
        if arg.startswith("--model="):
            candidates.append(arg.split("=", 1)[1])
        for candidate in candidates:
            if candidate == str(expected) or _resolve_process_arg(candidate, cwd) == expected_real:
                return True
    return False


def _cmdline_has_flag(cmdline: Sequence[str], name: str) -> bool:
    return name in cmdline


def _validate_runtime_contract_cmdline(
        cmdline: Sequence[str], runtime_contract: Mapping[str, Any]) -> None:
    for option, key in (
        (("--cache-type-k", "-ctk"), "cache_type_k"),
        (("--cache-type-v", "-ctv"), "cache_type_v"),
        (("--parallel", "-np"), "parallel"),
    ):
        actual = _cmdline_option_value(cmdline, set(option))
        expected = str(runtime_contract[key])
        if actual != expected:
            raise ServerProcessIdentityError(
                f"/proc/PID/cmdline {option[0]} differs from runtime contract"
            )
    for key, flag in (
        ("kv_unified", "--kv-unified"),
        ("no_cache_idle_slots", "--no-cache-idle-slots"),
        ("no_context_shift", "--no-context-shift"),
    ):
        expected = runtime_contract[key] is True
        if _cmdline_has_flag(cmdline, flag) != expected:
            raise ServerProcessIdentityError(
                f"/proc/PID/cmdline {flag} disagrees with runtime contract"
            )
    ctx_size = _cmdline_option_value(cmdline, {"--ctx-size"})
    if ctx_size != str(runtime_contract["ctx_size"]):
        raise ServerProcessIdentityError(
            "/proc/PID/cmdline --ctx-size differs from runtime contract"
        )


def verify_server_process_identity(
    authority: ServerAuthority,
    *,
    base_url: str,
    server_pid: int,
    model_path: str | os.PathLike[str],
    binary_path: str | os.PathLike[str],
    model_sha256: Optional[str] = None,
    binary_sha256: Optional[str] = None,
    runtime_contract: Optional[Mapping[str, Any]] = None,
    proc_root: str | os.PathLike[str] = "/proc",
) -> dict[str, Any]:
    """Bind the HTTP authority to one local llama-server process.

    User-supplied SHA values are optional consistency checks only.  The
    qualified values always come from the referenced files and /proc/PID.
    """
    if authority.source != "server_props_and_slots":
        raise ServerProcessIdentityError("process binding requires live server authority")
    if isinstance(server_pid, bool) or not isinstance(server_pid, int) or server_pid <= 0:
        raise ServerProcessIdentityError("real qualification requires a positive server PID")
    try:
        model_real = Path(model_path).expanduser().resolve(strict=True)
        binary_real = Path(binary_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ServerProcessIdentityError("model-path and binary-path must exist") from exc
    if not model_real.is_file() or not binary_real.is_file():
        raise ServerProcessIdentityError("model-path and binary-path must be regular files")
    actual_model_sha = sha256_file(model_real)
    actual_binary_sha = sha256_file(binary_real)
    if model_sha256 is not None and model_sha256 != actual_model_sha:
        raise ServerProcessIdentityError("provided model SHA does not match model-path")
    if binary_sha256 is not None and binary_sha256 != actual_binary_sha:
        raise ServerProcessIdentityError("provided binary SHA does not match binary-path")

    if not isinstance(authority.props_model_path, str) or not authority.props_model_path:
        raise ServerProcessIdentityError("/props did not publish a model_path")
    try:
        props_model_real = Path(authority.props_model_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ServerProcessIdentityError("/props model_path is not an existing file") from exc
    if os.path.realpath(str(props_model_real)) != os.path.realpath(str(model_real)):
        raise ServerProcessIdentityError(
            "running server /props model_path differs from requested model-path"
        )

    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "localhost", "127.0.0.1", "::1"
    }:
        raise ServerProcessIdentityError(
            "real process binding requires a local HTTP base-url"
        )
    try:
        base_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ServerProcessIdentityError("base-url has an invalid port") from exc

    proc_root_real = Path(proc_root).expanduser().resolve()
    proc_dir = proc_root_real / str(server_pid)
    try:
        proc_exe_link = os.readlink(proc_dir / "exe")
        proc_cwd = os.path.realpath(os.readlink(proc_dir / "cwd"))
    except OSError as exc:
        raise ServerProcessIdentityError(
            f"cannot read /proc/{server_pid}/exe or cwd"
        ) from exc
    proc_exe_real = Path(proc_exe_link).resolve(strict=True)
    if os.path.realpath(str(proc_exe_real)) != os.path.realpath(str(binary_real)):
        raise ServerProcessIdentityError("/proc/PID/exe differs from binary-path")
    proc_exe_sha = sha256_file(proc_exe_real)
    if proc_exe_sha != actual_binary_sha:
        raise ServerProcessIdentityError("/proc/PID/exe SHA differs from binary-path")
    cmdline = _read_proc_cmdline(proc_dir)
    if _resolve_process_arg(cmdline[0], proc_cwd) != os.path.realpath(str(binary_real)):
        raise ServerProcessIdentityError("/proc/PID/cmdline executable differs from binary-path")
    if not _cmdline_contains_path(cmdline, model_real, proc_cwd):
        raise ServerProcessIdentityError("/proc/PID/cmdline does not bind the model-path")
    port_text = _cmdline_option_value(cmdline, {"--port", "-p"})
    if port_text is None:
        raise ServerProcessIdentityError("/proc/PID/cmdline does not publish a server port")
    try:
        cmdline_port = int(port_text)
    except ValueError as exc:
        raise ServerProcessIdentityError("/proc/PID/cmdline port is invalid") from exc
    if cmdline_port != base_port:
        raise ServerProcessIdentityError(
            f"base-url port {base_port} differs from server cmdline port {cmdline_port}"
        )
    if runtime_contract is not None:
        _validate_runtime_contract_cmdline(cmdline, runtime_contract)
    proc_stat_starttime = _read_proc_stat_starttime(proc_dir)

    identity = {
        "pid": server_pid,
        "base_url": base_url,
        "port": base_port,
        "proc_exe": str(proc_exe_real),
        "proc_exe_sha256": proc_exe_sha,
        "proc_cwd": proc_cwd,
        "proc_root": str(proc_root_real),
        "proc_stat_starttime": proc_stat_starttime,
        "cmdline": list(cmdline),
        "model_path": str(model_real),
        "model_sha256": actual_model_sha,
        "binary_path": str(binary_real),
        "binary_sha256": actual_binary_sha,
        "props_model_path": authority.props_model_path,
    }
    identity["binding_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    return identity


def _current_process_starttime(identity: Mapping[str, Any], phase: str) -> str:
    pid = identity.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ServerProcessIdentityError(f"{phase} process identity has an invalid PID")
    proc_root = identity.get("proc_root", "/proc")
    if not isinstance(proc_root, str) or not proc_root:
        raise ServerProcessIdentityError(f"{phase} process identity has an invalid proc root")
    current = _read_proc_stat_starttime(Path(proc_root) / str(pid))
    expected = identity.get("proc_stat_starttime")
    if not isinstance(expected, str) or not expected.isdigit():
        raise ServerProcessIdentityError(
            f"{phase} process identity has no stable starttime authority"
        )
    if current != expected:
        raise ServerProcessIdentityError(
            f"{phase} server process starttime differs from bound identity"
        )
    return current


def capture_process_start_identity_snapshot(
    identity: Mapping[str, Any], phase: str
) -> dict[str, Any]:
    """Capture the loader-compatible PID/starttime snapshot for one phase.

    Keep the writer and loader on the same explicit schema.  A bare starttime
    string is intentionally never emitted because REAL transcript reload uses
    this snapshot as part of the process-identity trust boundary.
    """
    pid = identity.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ServerProcessIdentityError(f"{phase} process identity has an invalid PID")
    return {
        "pid": pid,
        "proc_stat_starttime": _current_process_starttime(identity, phase),
    }


def _validate_prompt_tokens(tokens: Sequence[int], label: str) -> tuple[int, ...]:
    result = tuple(tokens)
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in result):
        raise MaterializationError(f"{label} contains an illegal token id")
    return result


class SafeAlphabetMaterializer:
    """Deterministic PRF materializer with a collision-free session token."""

    def __init__(self, alphabet: Sequence[int]) -> None:
        self.alphabet = tuple(sorted(_validate_safe_alphabet(alphabet)))
        self._session_discriminators: dict[int, int] = {}

    def bind_session_ids(self, session_ids: Sequence[int]) -> None:
        normalized = tuple(sorted(set(int(value) for value in session_ids)))
        if len(normalized) != len(tuple(session_ids)):
            raise CalibrationError("session ids must be unique for discriminator binding")
        if len(self.alphabet) < len(normalized):
            raise CalibrationError(
                "safe alphabet has fewer legal non-special tokens than slice sessions"
            )
        proposed = {
            lineage_id: self.alphabet[index]
            for index, lineage_id in enumerate(normalized)
        }
        if self._session_discriminators and self._session_discriminators != proposed:
            raise CalibrationError("session discriminator binding changed after materialization")
        self._session_discriminators = proposed

    def materialize_block(self, lineage_id: int, block_id: str) -> tuple[int, ...]:
        if lineage_id not in self._session_discriminators:
            raise CalibrationError("materializer has no deterministic session discriminator binding")
        if not isinstance(block_id, str) or not block_id:
            raise MaterializationError("block id must be a non-empty string")
        match = _BLOCK_ID.fullmatch(block_id)
        if match is None or int(match.group(1)) != lineage_id:
            raise MaterializationError(
                f"block id {block_id!r} is not namespaced to lineage {lineage_id}"
            )
        seed = f"{MATERIALIZER_IDENTITY}|{lineage_id}|{block_id}".encode("utf-8")
        result: list[int] = []
        counter = 0
        while len(result) < BLOCK_SIZE_TOKENS:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for byte in digest:
                result.append(self.alphabet[byte % len(self.alphabet)])
                if len(result) == BLOCK_SIZE_TOKENS:
                    break
            counter += 1
        # Every block's first token carries the deterministic session identity.
        # This makes first-turn LCP=0 an invariant, not a PRF collision hope.
        result[0] = self._session_discriminators[lineage_id]
        return tuple(result)

    def materialize_blocks(
        self, lineage_id: int, block_ids: Sequence[str]
    ) -> tuple[int, ...]:
        output: list[int] = []
        for block_id in block_ids:
            output.extend(self.materialize_block(lineage_id, block_id))
        return tuple(output)

    def materialize_tail(
        self,
        lineage_id: int,
        turn: int,
        block_position: int,
        block_id: str,
        remainder: int,
    ) -> tuple[int, ...]:
        """Materialize the final, sub-block-remainder tokens of ONE turn.

        Per Alibaba trace semantics the last block of a turn may contain
        padding, and the first output token of the next turn replaces that
        padding; therefore a partial block's hash is NOT stable cross-request
        content identity.  This tail is intentionally request-local: it is
        deterministic, but its identity binds (lineage, turn, block position,
        block id, remainder) so two turns may legitimately materialize
        different remainder tokens at the same block position.  It never emits
        padding: only ``remainder`` legal tokens are produced.
        """
        if lineage_id not in self._session_discriminators:
            raise CalibrationError("materializer has no deterministic session discriminator binding")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
            raise MaterializationError("tail materialization turn is invalid")
        if (
            isinstance(block_position, bool)
            or not isinstance(block_position, int)
            or block_position < 0
        ):
            raise MaterializationError("tail materialization block position is invalid")
        if isinstance(remainder, bool) or not isinstance(remainder, int) or remainder <= 0:
            raise MaterializationError("tail materialization remainder must be positive")
        if remainder >= BLOCK_SIZE_TOKENS:
            raise MaterializationError(
                "tail materialization remainder must be smaller than the block size"
            )
        match = _BLOCK_ID.fullmatch(block_id)
        if match is None or int(match.group(1)) != lineage_id:
            raise MaterializationError(
                f"tail block id {block_id!r} is not namespaced to lineage {lineage_id}"
            )
        seed = (
            f"{MATERIALIZER_IDENTITY}|tail|{lineage_id}|{turn}|"
            f"{block_position}|{block_id}|{remainder}"
        ).encode("utf-8")
        result: list[int] = []
        counter = 0
        while len(result) < remainder:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for byte in digest:
                result.append(self.alphabet[byte % len(self.alphabet)])
                if len(result) == remainder:
                    break
            counter += 1
        # Every turn's first prompt token carries the deterministic session
        # identity.  A first turn may be entirely a partial tail (when its
        # input_length is below one block), so the tail must also own the
        # first-token discriminant — this keeps cross-session first-turn LCP=0
        # an invariant rather than a PRF collision hope.
        if block_position == 0:
            result[0] = self._session_discriminators[lineage_id]
        return tuple(result)

    def identity(self) -> dict[str, Any]:
        discriminator_items = [
            {"lineage_id": lineage_id, "token_id": token_id}
            for lineage_id, token_id in sorted(self._session_discriminators.items())
        ]
        return {
            "name": MATERIALIZER_IDENTITY,
            "block_size_tokens": BLOCK_SIZE_TOKENS,
            "safe_alphabet_sha256": sha256_bytes(
                canonical_json_bytes(list(self.alphabet))
            ),
            "session_discriminators": discriminator_items,
            "session_discriminators_sha256": sha256_bytes(
                canonical_json_bytes(discriminator_items)
            ),
        }


def _validate_safe_alphabet(alphabet: Sequence[int]) -> tuple[int, ...]:
    result = tuple(alphabet)
    if not result:
        raise CalibrationError("safe token alphabet must not be empty")
    if len(set(result)) != len(result):
        raise CalibrationError("safe token alphabet must be deterministic and unique")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in result):
        raise CalibrationError("safe token alphabet contains an illegal token id")
    return result


REFERENCE_GENERATION_SETTINGS: dict[str, Any] = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 0,
}


def build_reference_request(
    prompt_tokens: Sequence[int],
    n_predict: int,
    slot_id: int,
    *,
    cache_prompt: bool = True,
) -> dict[str, Any]:
    prompt = list(_validate_prompt_tokens(prompt_tokens, "reference prompt"))
    if isinstance(n_predict, bool) or not isinstance(n_predict, int) or n_predict < 0:
        raise MaterializationError("n_predict must be a non-negative integer")
    if isinstance(slot_id, bool) or not isinstance(slot_id, int) or slot_id < 0:
        raise MaterializationError("slot_id must be a non-negative integer")
    if not isinstance(cache_prompt, bool):
        raise MaterializationError("cache_prompt must be a boolean")
    request = {
        "prompt": prompt,
        "n_predict": n_predict,
        **REFERENCE_GENERATION_SETTINGS,
        "cache_prompt": cache_prompt,
        "return_tokens": True,
        "ignore_eos": True,
        "stream": False,
        "id_slot": slot_id,
    }
    # Deliberately no stop field: reference completion must not use a stop
    # string or silently fall back to a text-prompt path.
    return request


@dataclass(frozen=True)
class CompletionObservation:
    slot_id: int
    tokens_evaluated: int
    tokens_predicted: int
    tokens: tuple[int, ...]
    request_sha256: str
    response_sha256: str


def validate_completion_response(
    response: ApiResponse | Mapping[str, Any],
    *,
    expected_input_tokens: int,
    expected_n_predict: int,
    expected_slot_id: int,
) -> CompletionObservation:
    envelope = _as_response(response)
    payload = _require_mapping(envelope, "/completion reference")
    if payload.get("id_slot") != expected_slot_id:
        raise MaterializationError(
            "reference completion response slot identity mismatch"
        )
    tokens_evaluated = payload.get("tokens_evaluated")
    tokens_predicted = payload.get("tokens_predicted")
    if (
        isinstance(tokens_evaluated, bool)
        or not isinstance(tokens_evaluated, int)
        or tokens_evaluated != expected_input_tokens
    ):
        raise MaterializationError(
            "reference completion tokens_evaluated does not equal prompt length"
        )
    if (
        isinstance(tokens_predicted, bool)
        or not isinstance(tokens_predicted, int)
        or tokens_predicted != expected_n_predict
    ):
        raise MaterializationError(
            "reference completion tokens_predicted does not equal n_predict"
        )
    generated = payload.get("tokens")
    if not isinstance(generated, list) or len(generated) != expected_n_predict:
        raise MaterializationError(
            "reference completion returned the wrong number of completion tokens"
        )
    completion_tokens = _validate_prompt_tokens(generated, "reference completion")
    return CompletionObservation(
        slot_id=expected_slot_id,
        tokens_evaluated=tokens_evaluated,
        tokens_predicted=tokens_predicted,
        tokens=completion_tokens,
        request_sha256="",
        response_sha256=sha256_bytes(
            envelope.raw_body or canonical_json_bytes(payload)
        ),
    )


@dataclass(frozen=True)
class TurnTranscript:
    lineage_id: int
    turn: int
    timestamp: float
    input_length: int
    n_predict: int
    type: str
    block_ids: tuple[str, ...]
    prompt_tokens: tuple[int, ...]
    reference_completion_tokens: tuple[int, ...]
    prompt_sha256: str
    reference_completion_sha256: str
    request_sha256: str
    response_sha256: str
    slot_id: int
    tokens_evaluated: int
    tokens_predicted: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "turn": self.turn,
            "timestamp": self.timestamp,
            "input_length": self.input_length,
            "n_predict": self.n_predict,
            "type": self.type,
            "block_ids": list(self.block_ids),
            "prompt_tokens": list(self.prompt_tokens),
            "reference_completion_tokens": list(self.reference_completion_tokens),
            "prompt_sha256": self.prompt_sha256,
            "reference_completion_sha256": self.reference_completion_sha256,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "slot_id": self.slot_id,
            "tokens_evaluated": self.tokens_evaluated,
            "tokens_predicted": self.tokens_predicted,
        }


@dataclass
class TokenTranscript:
    schema_version: str
    materialize_mode: str
    materialization_status: str
    parent: dict[str, Any]
    model: dict[str, Any]
    effective_n_ctx_authority: dict[str, Any]
    token_calibration: dict[str, Any]
    materializer: dict[str, Any]
    reference_pass: dict[str, Any]
    turns: list[TurnTranscript]
    runtime_contract: dict[str, Any] | None = None
    transcript_sha256: str = ""

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "materialize_mode": self.materialize_mode,
            "materialization_status": self.materialization_status,
            "parent": self.parent,
            "model": self.model,
            "effective_n_ctx_authority": self.effective_n_ctx_authority,
            "token_calibration": self.token_calibration,
            "materializer": self.materializer,
            "reference_pass": self.reference_pass,
            "turns": [turn.to_dict() for turn in self.turns],
            "runtime_contract": self.runtime_contract,
        }

    def finalize(self) -> "TokenTranscript":
        self.transcript_sha256 = sha256_bytes(canonical_json_bytes(self.unsigned_dict()))
        return self

    def to_dict(self) -> dict[str, Any]:
        if not self.transcript_sha256:
            self.finalize()
        result = self.unsigned_dict()
        result["transcript_sha256"] = self.transcript_sha256
        return result

    def to_canonical_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    def write(self, path: str | os.PathLike[str]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_canonical_json() + "\n", encoding="utf-8")


def _strict_int(value: Any, label: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MaterializationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise MaterializationError(f"{label} is below the minimum")
    return value


def _strict_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterializationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise MaterializationError(f"{label} must be finite")
    return result


def _strict_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise MaterializationError(f"{label} must be a 64-hex SHA256")
    return value


def _strict_process_identity(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MaterializationError(f"{label} must be an object")
    required = {
        "pid",
        "base_url",
        "port",
        "proc_exe",
        "proc_exe_sha256",
        "proc_cwd",
        "proc_root",
        "proc_stat_starttime",
        "cmdline",
        "model_path",
        "model_sha256",
        "binary_path",
        "binary_sha256",
        "props_model_path",
        "binding_sha256",
    }
    if set(raw) != required:
        raise MaterializationError(f"{label} schema mismatch")
    _strict_int(raw["pid"], f"{label}.pid", minimum=1)
    _strict_int(raw["port"], f"{label}.port", minimum=1)
    for name in (
        "base_url",
        "proc_exe",
        "proc_cwd",
        "proc_root",
        "model_path",
        "binary_path",
        "props_model_path",
    ):
        if not isinstance(raw[name], str) or not raw[name]:
            raise MaterializationError(f"{label}.{name} must be a non-empty string")
    starttime = raw["proc_stat_starttime"]
    if not isinstance(starttime, str) or not starttime.isdigit():
        raise MaterializationError(f"{label}.proc_stat_starttime is invalid")
    cmdline = raw["cmdline"]
    if (
        not isinstance(cmdline, list)
        or not cmdline
        or any(not isinstance(value, str) or not value for value in cmdline)
    ):
        raise MaterializationError(f"{label}.cmdline is invalid")
    for name in (
        "proc_exe_sha256",
        "model_sha256",
        "binary_sha256",
        "binding_sha256",
    ):
        _strict_sha(raw[name], f"{label}.{name}")
    unsigned = dict(raw)
    binding_sha = unsigned.pop("binding_sha256")
    if sha256_bytes(canonical_json_bytes(unsigned)) != binding_sha:
        raise MaterializationError(f"{label}.binding_sha256 is inconsistent")
    return dict(raw)


def _strict_start_snapshot(raw: Any, label: str) -> Optional[dict[str, Any]]:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != {"pid", "proc_stat_starttime"}:
        raise MaterializationError(f"{label} is invalid")
    _strict_int(raw["pid"], f"{label}.pid", minimum=1)
    if not isinstance(raw["proc_stat_starttime"], str) or not raw["proc_stat_starttime"].isdigit():
        raise MaterializationError(f"{label}.proc_stat_starttime is invalid")
    return dict(raw)


def _load_turn_transcript(raw: Mapping[str, Any], index: int) -> TurnTranscript:
    if not isinstance(raw, Mapping):
        raise MaterializationError(f"transcript turn {index} must be an object")
    required = {
        "lineage_id",
        "turn",
        "timestamp",
        "input_length",
        "n_predict",
        "type",
        "block_ids",
        "prompt_tokens",
        "reference_completion_tokens",
        "prompt_sha256",
        "reference_completion_sha256",
        "request_sha256",
        "response_sha256",
        "slot_id",
        "tokens_evaluated",
        "tokens_predicted",
    }
    if set(raw) != required:
        raise MaterializationError(f"transcript turn {index} schema mismatch")
    lineage_id = _strict_int(raw["lineage_id"], f"transcript turn {index}.lineage_id", minimum=0)
    turn_number = _strict_int(raw["turn"], f"transcript turn {index}.turn", minimum=1)
    timestamp = _strict_float(raw["timestamp"], f"transcript turn {index}.timestamp")
    input_length = _strict_int(raw["input_length"], f"transcript turn {index}.input_length", minimum=0)
    n_predict = _strict_int(raw["n_predict"], f"transcript turn {index}.n_predict", minimum=0)
    slot_id = _strict_int(raw["slot_id"], f"transcript turn {index}.slot_id", minimum=0)
    tokens_evaluated = _strict_int(
        raw["tokens_evaluated"], f"transcript turn {index}.tokens_evaluated", minimum=0
    )
    tokens_predicted = _strict_int(
        raw["tokens_predicted"], f"transcript turn {index}.tokens_predicted", minimum=0
    )
    if not isinstance(raw["type"], str) or not raw["type"]:
        raise MaterializationError(f"transcript turn {index}.type is invalid")
    block_ids = raw["block_ids"]
    if not isinstance(block_ids, list) or any(
        not isinstance(value, str) or not value for value in block_ids
    ):
        raise MaterializationError(f"transcript turn {index}.block_ids is invalid")
    for block_id in block_ids:
        match = _BLOCK_ID.fullmatch(block_id)
        if match is None or int(match.group(1)) != lineage_id:
            raise MaterializationError(f"transcript turn {index}.block_ids is not namespaced")
    token_lists: dict[str, tuple[int, ...]] = {}
    for name in ("prompt_tokens", "reference_completion_tokens"):
        values = raw[name]
        if not isinstance(values, list):
            raise MaterializationError(f"transcript turn {index} {name} is not a list")
        token_lists[name] = _validate_prompt_tokens(values, f"transcript turn {index} {name}")
    prompt_tokens = token_lists["prompt_tokens"]
    completion_tokens = token_lists["reference_completion_tokens"]
    if len(prompt_tokens) != input_length:
        raise MaterializationError(f"transcript turn {index} prompt length mismatch")
    if len(completion_tokens) != n_predict:
        raise MaterializationError(f"transcript turn {index} completion length mismatch")
    if tokens_evaluated != input_length:
        raise MaterializationError(f"transcript turn {index} evaluated count mismatch")
    if tokens_predicted != n_predict:
        raise MaterializationError(f"transcript turn {index} predicted count mismatch")
    prompt_sha = _strict_sha(raw["prompt_sha256"], f"transcript turn {index}.prompt_sha256")
    completion_sha = _strict_sha(
        raw["reference_completion_sha256"],
        f"transcript turn {index}.reference_completion_sha256",
    )
    request_sha = _strict_sha(raw["request_sha256"], f"transcript turn {index}.request_sha256")
    response_sha = _strict_sha(raw["response_sha256"], f"transcript turn {index}.response_sha256")
    if prompt_sha != sha256_bytes(canonical_json_bytes(list(prompt_tokens))):
        raise MaterializationError(f"transcript turn {index} prompt token SHA mismatch")
    if completion_sha != sha256_bytes(canonical_json_bytes(list(completion_tokens))):
        raise MaterializationError(f"transcript turn {index} completion token SHA mismatch")
    return TurnTranscript(
        lineage_id=lineage_id,
        turn=turn_number,
        timestamp=timestamp,
        input_length=input_length,
        n_predict=n_predict,
        type=raw["type"],
        block_ids=tuple(block_ids),
        prompt_tokens=prompt_tokens,
        reference_completion_tokens=completion_tokens,
        prompt_sha256=prompt_sha,
        reference_completion_sha256=completion_sha,
        request_sha256=request_sha,
        response_sha256=response_sha,
        slot_id=slot_id,
        tokens_evaluated=tokens_evaluated,
        tokens_predicted=tokens_predicted,
    )


def load_transcript(path: str | os.PathLike[str]) -> TokenTranscript:
    """Reload a frozen transcript without touching the source trace."""
    raw = Path(path).read_bytes()
    document = _strict_json_object(raw, str(path))
    required = {
        "schema_version",
        "materialize_mode",
        "materialization_status",
        "parent",
        "model",
        "effective_n_ctx_authority",
        "token_calibration",
        "materializer",
        "reference_pass",
        "turns",
        "runtime_contract",
        "transcript_sha256",
    }
    if set(document) != required:
        raise MaterializationError("transcript top-level schema mismatch")
    expected_sha = _strict_sha(document["transcript_sha256"], "transcript_sha256")
    unsigned = dict(document)
    unsigned.pop("transcript_sha256")
    actual_sha = sha256_bytes(canonical_json_bytes(unsigned))
    if actual_sha != expected_sha:
        raise MaterializationError(
            f"transcript SHA mismatch: expected={expected_sha} actual={actual_sha}"
        )
    if document["schema_version"] != TRANSCRIPT_SCHEMA_VERSION:
        raise MaterializationError("unsupported token transcript schema version")
    mode = document["materialize_mode"]
    status = document["materialization_status"]
    if not isinstance(mode, str) or not isinstance(status, str):
        raise MaterializationError("transcript materialization state must be strings")
    for name in (
        "parent",
        "model",
        "effective_n_ctx_authority",
        "token_calibration",
        "materializer",
        "reference_pass",
    ):
        if not isinstance(document[name], Mapping):
            raise MaterializationError(f"transcript {name} must be an object")
    runtime_contract: dict[str, Any] | None
    if document["runtime_contract"] is None:
        runtime_contract = None
    else:
        runtime_contract = _strict_runtime_contract(document["runtime_contract"], "transcript.runtime_contract")

    model = dict(document["model"])
    if set(model) != {"model_sha256", "binary_sha256", "identity_status", "server_process_identity"}:
        raise MaterializationError("transcript model schema mismatch")
    identity_status = model["identity_status"]
    if identity_status not in {"REAL", "UNVERIFIED"}:
        raise MaterializationError("transcript model identity_status is invalid")
    if mode == MATERIALIZE_MODE_REAL or status == MATERIALIZATION_STATUS_REAL or identity_status == "REAL":
        if (mode, status, identity_status) != (
            MATERIALIZE_MODE_REAL,
            MATERIALIZATION_STATUS_REAL,
            "REAL",
        ):
            raise MaterializationError("REAL transcript state is not mutually consistent")
        is_real = True
    elif (mode, status, identity_status) == (
        MATERIALIZE_MODE_OPEN,
        MATERIALIZATION_STATUS_UNVERIFIED,
        "UNVERIFIED",
    ):
        is_real = False
    else:
        raise MaterializationError("transcript OPEN/UNVERIFIED state is not mutually consistent")

    turns_raw = document["turns"]
    if not isinstance(turns_raw, list):
        raise MaterializationError("transcript turns must be a list")
    turns = [_load_turn_transcript(item, index) for index, item in enumerate(turns_raw, 1)]
    by_session: dict[int, list[TurnTranscript]] = {}
    for turn in turns:
        by_session.setdefault(turn.lineage_id, []).append(turn)
    for lineage_id, session_turns in by_session.items():
        if [item.turn for item in session_turns] != list(range(1, len(session_turns) + 1)):
            raise MaterializationError(f"transcript lineage {lineage_id} is not contiguous")
        for previous, current in zip(session_turns, session_turns[1:]):
            expected_lcp = expected_full_block_lcp(previous, current)
            if (
                current.prompt_tokens[:expected_lcp] != previous.prompt_tokens[:expected_lcp]
            ):
                raise MaterializationError(
                    f"transcript lineage {lineage_id} turn {current.turn} violates "
                    f"the expected_full_block_lcp reuse contract "
                    f"(expected {expected_lcp} shared full-block prefix tokens)"
                )
    _check_cross_session_lcp(turns)

    calibration = dict(document["token_calibration"])
    calibration_required = {
        "corpus",
        "repetitions",
        "request_sha256",
        "response_sha256",
        "safe_alphabet",
        "safe_alphabet_sha256",
        "model_sha256",
        "binary_sha256",
        "real_model",
        "special_token_ids",
    }
    if set(calibration) != calibration_required:
        raise MaterializationError("transcript token_calibration schema mismatch")
    if (
        not isinstance(calibration["corpus"], list)
        or any(not isinstance(value, str) or not value for value in calibration["corpus"])
    ):
        raise MaterializationError("transcript calibration corpus is invalid")
    _strict_int(calibration["repetitions"], "token_calibration.repetitions", minimum=2)
    _strict_sha(calibration["request_sha256"], "token_calibration.request_sha256")
    _strict_sha(calibration["response_sha256"], "token_calibration.response_sha256")
    safe_alphabet = calibration["safe_alphabet"]
    if not isinstance(safe_alphabet, list):
        raise MaterializationError("token_calibration.safe_alphabet is invalid")
    try:
        safe_alphabet_tuple = _validate_safe_alphabet(safe_alphabet)
    except MaterializationError as exc:
        raise MaterializationError("token_calibration.safe_alphabet is invalid") from exc
    if calibration["safe_alphabet_sha256"] != sha256_bytes(
        canonical_json_bytes(list(safe_alphabet_tuple))
    ):
        raise MaterializationError("token_calibration.safe_alphabet_sha256 is inconsistent")
    for name in ("model_sha256", "binary_sha256"):
        value = calibration[name]
        if value is not None:
            _strict_sha(value, f"token_calibration.{name}")
    if not isinstance(calibration["real_model"], bool):
        raise MaterializationError("token_calibration.real_model must be boolean")
    special_ids = calibration["special_token_ids"]
    if not isinstance(special_ids, list):
        raise MaterializationError("token_calibration.special_token_ids is invalid")
    for value in special_ids:
        _strict_int(value, "token_calibration.special_token_ids entry", minimum=0)

    authority = dict(document["effective_n_ctx_authority"])
    authority_required = {
        "source",
        "slot_id",
        "props_n_ctx",
        "slot_n_ctx",
        "effective_n_ctx",
        "props_sha256",
        "slots_sha256",
        "probe_request_sha256",
        "probe_response_sha256",
        "contract_version",
        "props_model_path",
        "process_identity",
    }
    if set(authority) != authority_required:
        raise MaterializationError("transcript authority schema mismatch")
    if not isinstance(authority["source"], str) or not authority["source"]:
        raise MaterializationError("transcript authority source is invalid")
    for name in ("slot_id", "props_n_ctx", "slot_n_ctx", "effective_n_ctx"):
        _strict_int(authority[name], f"authority.{name}", minimum=0 if name == "slot_id" else 1)
    for name in ("props_sha256", "slots_sha256", "probe_request_sha256", "probe_response_sha256"):
        value = authority[name]
        if value != "UNVERIFIED":
            _strict_sha(value, f"authority.{name}")
    if not isinstance(authority["contract_version"], str) or not authority["contract_version"]:
        raise MaterializationError("transcript authority contract_version is invalid")
    if authority["props_model_path"] is not None and (
        not isinstance(authority["props_model_path"], str) or not authority["props_model_path"]
    ):
        raise MaterializationError("transcript authority props_model_path is invalid")
    authority_process = authority["process_identity"]
    if authority_process is not None:
        authority_process = _strict_process_identity(authority_process, "authority.process_identity")

    materializer = dict(document["materializer"])
    materializer_required = {
        "name",
        "block_size_tokens",
        "safe_alphabet_sha256",
        "session_discriminators",
        "session_discriminators_sha256",
    }
    if set(materializer) != materializer_required:
        raise MaterializationError("transcript materializer schema mismatch")
    if materializer["name"] != MATERIALIZER_IDENTITY:
        raise MaterializationError("transcript materializer identity is unsupported")
    if materializer["block_size_tokens"] != BLOCK_SIZE_TOKENS:
        raise MaterializationError("transcript materializer block size mismatch")
    if materializer["safe_alphabet_sha256"] != calibration["safe_alphabet_sha256"]:
        raise MaterializationError("transcript materializer alphabet hash mismatch")
    discriminators = materializer["session_discriminators"]
    if not isinstance(discriminators, list):
        raise MaterializationError("transcript session discriminators are invalid")
    normalized_discriminators: list[dict[str, int]] = []
    for item in discriminators:
        if not isinstance(item, Mapping) or set(item) != {"lineage_id", "token_id"}:
            raise MaterializationError("transcript session discriminator schema mismatch")
        normalized_discriminators.append({
            "lineage_id": _strict_int(item["lineage_id"], "session discriminator lineage_id", minimum=0),
            "token_id": _strict_int(item["token_id"], "session discriminator token_id", minimum=0),
        })
    if normalized_discriminators != sorted(normalized_discriminators, key=lambda item: item["lineage_id"]):
        raise MaterializationError("transcript session discriminators are not sorted")
    if len({item["lineage_id"] for item in normalized_discriminators}) != len(normalized_discriminators):
        raise MaterializationError("transcript session discriminators repeat a lineage")
    if len({item["token_id"] for item in normalized_discriminators}) != len(normalized_discriminators):
        raise MaterializationError("transcript session discriminators repeat a token")
    if {item["lineage_id"] for item in normalized_discriminators} != set(by_session):
        raise MaterializationError("transcript session discriminators do not cover all sessions")
    if any(item["token_id"] not in safe_alphabet_tuple for item in normalized_discriminators):
        raise MaterializationError("transcript session discriminator is outside safe alphabet")
    if materializer["session_discriminators_sha256"] != sha256_bytes(
        canonical_json_bytes(normalized_discriminators)
    ):
        raise MaterializationError("transcript session discriminator hash is inconsistent")

    reference_pass = dict(document["reference_pass"])
    reference_required = {
        "name",
        "request_count",
        "completed",
        "request_sha256",
        "response_sha256",
        "generation_settings",
        "execution_order",
        "artifact_order",
        "reference_completion_role",
        "formal_correctness_oracle",
        "reuse_contract",
        "block_lcp_observations",
        "fixed_request_fields",
        "process_start_identity_before",
        "process_start_identity_after",
    }
    if set(reference_pass) != reference_required:
        raise MaterializationError("transcript reference_pass schema mismatch")
    if reference_pass["name"] != REFERENCE_PASS_NAME:
        raise MaterializationError("transcript reference_pass name is invalid")
    if _strict_int(reference_pass["request_count"], "reference_pass.request_count", minimum=0) != len(turns):
        raise MaterializationError("reference_pass request_count does not match turns")
    if reference_pass["completed"] is not True:
        raise MaterializationError("reference_pass is not completed")
    _strict_sha(reference_pass["request_sha256"], "reference_pass.request_sha256")
    _strict_sha(reference_pass["response_sha256"], "reference_pass.response_sha256")
    if reference_pass["generation_settings"] != REFERENCE_GENERATION_SETTINGS:
        raise MaterializationError("reference_pass generation settings mismatch")
    if reference_pass["execution_order"] != "session_contiguous":
        raise MaterializationError("reference_pass execution order is not session-contiguous")
    if reference_pass["artifact_order"] != "parent_event_order":
        raise MaterializationError("reference_pass artifact order is not parent-event order")
    if reference_pass["reference_completion_role"] != REFERENCE_COMPLETION_ROLE:
        raise MaterializationError("reference completion role is invalid")
    if reference_pass["formal_correctness_oracle"] != FORMAL_CORRECTNESS_ORACLE:
        raise MaterializationError("reference formal oracle declaration is invalid")
    if reference_pass["reuse_contract"] != "expected_full_block_lcp":
        raise MaterializationError("reference_pass reuse contract declaration is invalid")
    block_lcp_observations = reference_pass["block_lcp_observations"]
    if (
        not isinstance(block_lcp_observations, list)
        or not all(isinstance(item, Mapping) for item in block_lcp_observations)
    ):
        raise MaterializationError("reference_pass block_lcp_observations is invalid")
    for entry in block_lcp_observations:
        if set(entry) != {
            "lineage_id",
            "from_turn",
            "to_turn",
            "expected_full_block_lcp",
            "actual_token_lcp",
        }:
            raise MaterializationError("reference_pass block_lcp_observations entry schema mismatch")
        _strict_int(entry["lineage_id"], "block_lcp_observations.lineage_id", minimum=0)
        if _strict_int(entry["from_turn"], "block_lcp_observations.from_turn", minimum=1) != (
            _strict_int(entry["to_turn"], "block_lcp_observations.to_turn", minimum=1) - 1
        ):
            raise MaterializationError(
                "block_lcp_observations adjacency is not from_turn = to_turn - 1"
            )
        expected_lcp = _strict_int(
            entry["expected_full_block_lcp"],
            "block_lcp_observations.expected_full_block_lcp",
            minimum=0,
        )
        if expected_lcp % BLOCK_SIZE_TOKENS != 0:
            raise MaterializationError(
                "block_lcp_observations.expected_full_block_lcp is not a block multiple"
            )
        _strict_int(entry["actual_token_lcp"], "block_lcp_observations.actual_token_lcp", minimum=0)
    fixed = reference_pass["fixed_request_fields"]
    if not isinstance(fixed, Mapping) or set(fixed) != {
        "cache_prompt",
        "cold_start_cache_prompt",
        "continuation_cache_prompt",
        "return_tokens",
        "ignore_eos",
        "stream",
        "stop",
        "prompt_kind",
    }:
        raise MaterializationError("reference_pass fixed request fields are invalid")
    if fixed != {
        "cache_prompt": True,
        "cold_start_cache_prompt": False,
        "continuation_cache_prompt": True,
        "return_tokens": True,
        "ignore_eos": True,
        "stream": False,
        "stop": "absent",
        "prompt_kind": "direct_token_ids",
    }:
        raise MaterializationError("reference_pass fixed request fields mismatch")
    start_before = _strict_start_snapshot(
        reference_pass["process_start_identity_before"],
        "reference_pass.process_start_identity_before",
    )
    start_after = _strict_start_snapshot(
        reference_pass["process_start_identity_after"],
        "reference_pass.process_start_identity_after",
    )

    model_sha = model["model_sha256"]
    binary_sha = model["binary_sha256"]
    if model_sha is not None:
        _strict_sha(model_sha, "model.model_sha256")
    if binary_sha is not None:
        _strict_sha(binary_sha, "model.binary_sha256")
    model_process = model["server_process_identity"]
    if model_process is not None:
        model_process = _strict_process_identity(model_process, "model.server_process_identity")
    if is_real:
        if runtime_contract is None:
            raise MaterializationError("REAL transcript runtime contract is missing")
        if calibration["real_model"] is not True:
            raise MaterializationError("REAL transcript calibration is not marked real_model")
        if not isinstance(model_sha, str) or not isinstance(binary_sha, str):
            raise MaterializationError("REAL transcript model/binary identity is incomplete")
        if model_sha != calibration["model_sha256"] or binary_sha != calibration["binary_sha256"]:
            raise MaterializationError("REAL transcript model/binary SHA differs from calibration")
        if authority["source"] != "server_props_and_slots" or authority_process is None:
            raise MaterializationError("REAL transcript lacks live authority process identity")
        if model_process != authority_process:
            raise MaterializationError("REAL transcript model and authority process identities differ")
        if authority_process["model_sha256"] != model_sha or authority_process["binary_sha256"] != binary_sha:
            raise MaterializationError("REAL transcript process SHA differs from model identity")
        if authority["props_model_path"] != authority_process["props_model_path"]:
            raise MaterializationError("REAL transcript /props model path differs from process identity")
        expected_start_snapshot = {
            "pid": authority_process["pid"],
            "proc_stat_starttime": authority_process["proc_stat_starttime"],
        }
        if (
            start_before is None
            or start_after is None
            or start_before != start_after
            or start_before != expected_start_snapshot
        ):
            raise MaterializationError(
                "REAL transcript process start identity is not stable or bound to authority"
            )
    else:
        if runtime_contract is not None:
            raise MaterializationError("OPEN transcript cannot carry runtime contract")
        if calibration["real_model"] is not False:
            raise MaterializationError("OPEN transcript cannot carry real calibration")
        if model_sha is not None or binary_sha is not None or model_process is not None:
            raise MaterializationError("OPEN transcript contains real model identity")
        if authority_process is not None or start_before is not None or start_after is not None:
            raise MaterializationError("OPEN transcript contains real process identity")

    # Per-turn full-block materialization identity: re-derive the stable
    # synthetic content of each turn's FULL blocks and require the recorded
    # prompt to match exactly.  Partial tail blocks are deliberately NOT
    # checked here, because a partial block's hash is request-local identity
    # and may legitimately differ across turns; the prompt_sha256 already
    # guards the recorded tail bytes against silent corruption.  This is the
    # load-side half of the Alibaba reuse contract: the same full block id
    # must always have produced the same 16-token sequence, regardless of any
    # transcript_sha256 recompute on a tampered block.
    replay_materializer = SafeAlphabetMaterializer(safe_alphabet_tuple)
    replay_materializer.bind_session_ids(
        tuple(item["lineage_id"] for item in normalized_discriminators)
    )
    for item in normalized_discriminators:
        if replay_materializer._session_discriminators[item["lineage_id"]] != item["token_id"]:
            raise MaterializationError("transcript materializer discriminators cannot be reproduced")
    for turn in turns:
        full_count = turn.input_length // BLOCK_SIZE_TOKENS
        if full_count == 0:
            continue
        if len(turn.block_ids) < full_count:
            raise MaterializationError(
                f"transcript turn {turn.turn} block list cannot cover its full-block prefix"
            )
        for position in range(full_count):
            block_id = turn.block_ids[position]
            expected_block = replay_materializer.materialize_block(turn.lineage_id, block_id)
            recorded = turn.prompt_tokens[position * BLOCK_SIZE_TOKENS : (position + 1) * BLOCK_SIZE_TOKENS]
            if recorded != expected_block:
                raise MaterializationError(
                    f"transcript lineage {turn.lineage_id} turn {turn.turn} full block {block_id} "
                    f"at position {position} does not reproduce its declared synthetic content"
                )

    return TokenTranscript(
        schema_version=document["schema_version"],
        materialize_mode=mode,
        materialization_status=status,
        parent=dict(document["parent"]),
        model=model,
        effective_n_ctx_authority=authority,
        token_calibration=calibration,
        materializer=materializer,
        reference_pass=reference_pass,
        turns=turns,
        runtime_contract=runtime_contract,
        transcript_sha256=expected_sha,
    )


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _validate_context(authority: ServerAuthority, input_length: int, n_predict: int) -> None:
    required = input_length + n_predict
    if required > authority.effective_n_ctx:
        raise ContextOverflowError(
            f"input_length+n_predict={required} exceeds authoritative "
            f"effective_n_ctx={authority.effective_n_ctx}"
        )


def expected_full_block_lcp(
    previous: TurnTranscript,
    current: FrozenTurn,
) -> int:
    """Same-session adjacent-turn reuse contract in tokens.

    Alibaba trace verifies (docs/qa-context-growth-pattern.md) that complete
    16-token blocks keep stable synthetic content across requests.  The runtime
    reuse invariant between two adjacent turns of the same lineage is therefore
    ``expected_full_block_lcp`` tokens: walk both block lists in lockstep, and
    for every block position below BOTH turns' full-block prefix lengths whose
    ``block_id`` matches, add ``BLOCK_SIZE_TOKENS``.  The first mismatched full
    block, or the first position that is partial in either turn, stops the
    count.  Partial blocks are deliberately excluded: the tail block's hash is
    not stable cross-request content identity.
    """
    prev_full = previous.input_length // BLOCK_SIZE_TOKENS
    curr_full = current.input_length // BLOCK_SIZE_TOKENS
    shared_block_cap = min(prev_full, curr_full)
    token_lcp = 0
    for position in range(shared_block_cap):
        if position >= len(previous.block_ids) or position >= len(current.block_ids):
            break
        if previous.block_ids[position] != current.block_ids[position]:
            break
        token_lcp += BLOCK_SIZE_TOKENS
    return token_lcp


def _build_prompt(
    current: FrozenTurn,
    materializer: SafeAlphabetMaterializer,
) -> tuple[int, ...]:
    """Materialize one turn's prompt directly from its own frozen constraint.

    Each turn's prompt is reconstructed from its own ``input_length`` and
    ``session_namespaced_block_ids`` only, exactly as the Alibaba
    Trace-Replayer does for stable synthetic content.  It never borrows the
    previous turn's reference completion: that completion is qualification
    observation only and may have had special tokens stripped by the next
    turn's input, so it is not a strict prefix invariant.
    """
    input_length = current.input_length
    full_count = input_length // BLOCK_SIZE_TOKENS
    remainder = input_length % BLOCK_SIZE_TOKENS
    prompt: list[int] = []
    if full_count > 0:
        if len(current.block_ids) < full_count:
            raise ParentArtifactError(
                "current block constraint cannot reach the full-block prefix"
            )
        prompt.extend(
            materializer.materialize_blocks(
                current.lineage_id, current.block_ids[:full_count]
            )
        )
    if remainder > 0:
        partial_position = full_count
        if partial_position >= len(current.block_ids):
            raise ParentArtifactError(
                "current block constraint cannot cover the partial tail"
            )
        prompt.extend(
            materializer.materialize_tail(
                current.lineage_id,
                current.turn,
                partial_position,
                current.block_ids[partial_position],
                remainder,
            )
        )
    if len(prompt) != input_length:
        raise MaterializationError(
            "per-turn prompt construction did not reach exact input length"
        )
    return tuple(prompt)


def _completion_request(client: ServerClient, request: Mapping[str, Any]) -> ApiResponse:
    if "prompt" not in request or not isinstance(request["prompt"], list):
        raise DirectTokenContractError("completion request must use a direct integer prompt list")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in request["prompt"]):
        raise DirectTokenContractError("completion request prompt is not an integer-token list")
    return _as_response(client.post("/completion", request))


def _check_cross_session_lcp(turns: Sequence[TurnTranscript]) -> None:
    first_turns = [turn for turn in turns if turn.turn == 1]
    for index, left in enumerate(first_turns):
        for right in first_turns[index + 1 :]:
            lcp = _longest_common_prefix(left.prompt_tokens, right.prompt_tokens)
            if lcp > 0:
                raise CrossSessionIsolationError(
                    f"sessions {left.lineage_id} and {right.lineage_id} share "
                    f"a first-turn prompt prefix of length {lcp}"
                )


def materialize_reference_transcript(
    frozen: FrozenSlice,
    calibration: TokenCalibration,
    client: ServerClient,
    *,
    authority: Optional[ServerAuthority] = None,
    slot_id: int = 0,
    materializer: Optional[SafeAlphabetMaterializer] = None,
    runtime_contract: Optional[dict[str, Any]] = None,
) -> TokenTranscript:
    """Run exactly one sequential Resident reference pass and freeze tokens."""
    if not isinstance(calibration, TokenCalibration):
        raise CalibrationError("reference materialization requires TokenCalibration")
    if calibration.real_model:
        if runtime_contract is None:
            raise DirectTokenContractError("real materialization requires runtime contract")
        runtime_contract = _strict_runtime_contract(runtime_contract, "runtime_contract")
    elif runtime_contract is not None:
        raise DirectTokenContractError("OPEN materialization cannot carry runtime contract")
    materializer = materializer or SafeAlphabetMaterializer(calibration.safe_alphabet)
    if authority is None:
        authority = preflight_direct_token_contract(
            client, probe_token_id=calibration.safe_alphabet[0], slot_id=slot_id
        )
    materializer.bind_session_ids(frozen.session_ids)
    if authority.slot_id != slot_id:
        raise DirectTokenContractError("reference slot differs from preflight authority")
    if calibration.real_model and authority.source != "server_props_and_slots":
        raise DirectTokenContractError(
            "real materialization requires /props + /slots context authority"
        )
    if calibration.real_model:
        _validate_identity_for_real(calibration.model_sha256, calibration.binary_sha256)
        if not authority.process_identity:
            raise ServerProcessIdentityError(
                "real materialization requires bound server process identity"
            )
    process_start_before: Optional[dict[str, Any]] = None
    if calibration.real_model:
        assert authority.process_identity is not None
        process_start_before = capture_process_start_identity_snapshot(
            authority.process_identity, "reference pass before"
        )

    turns_by_session: dict[int, list[FrozenTurn]] = {}
    for current in frozen.turns:
        turns_by_session.setdefault(current.lineage_id, []).append(current)
    executed_by_key: dict[tuple[int, int], TurnTranscript] = {}
    request_hashes: list[str] = []
    response_hashes: list[str] = []
    # Same-session reuse observations: for each adjacent pair, record expected
    # block-LCP and the actual token LCP.  The expected value is part of the
    # trust boundary; the actual value is a 1B-B fidelity observation only.
    block_lcp_observations: list[dict[str, Any]] = []
    for lineage_id in sorted(turns_by_session):
        previous: Optional[TurnTranscript] = None
        for current in turns_by_session[lineage_id]:
            _validate_context(authority, current.input_length, current.n_predict)
            prompt_tokens = _build_prompt(current, materializer)
            if previous is not None:
                expected_lcp = expected_full_block_lcp(previous, current)
                actual_lcp = _longest_common_prefix(
                    previous.prompt_tokens, prompt_tokens
                )
                # The runtime reuse contract is exact: the materialized
                # prompt must agree on the full-block prefix that the shared
                # block ids declare.  A divergence here means the frozen block
                # constraint was not honored.
                if prompt_tokens[:expected_lcp] != previous.prompt_tokens[:expected_lcp]:
                    raise PartialBlockConflictError(
                        f"lineage {lineage_id} turn {current.turn} reuses block "
                        f"ids {current.block_ids[:expected_lcp//BLOCK_SIZE_TOKENS]} "
                        f"but the materialized full-block prefix differs "
                        f"from turn {previous.turn}"
                    )
                block_lcp_observations.append({
                    "lineage_id": lineage_id,
                    "from_turn": previous.turn,
                    "to_turn": current.turn,
                    "expected_full_block_lcp": expected_lcp,
                    "actual_token_lcp": actual_lcp,
                })
            # cache_prompt=false is the native cold-start/reset boundary between
            # sessions; only a same-session continuation may reuse slot KV.
            request = build_reference_request(
                prompt_tokens,
                current.n_predict,
                slot_id,
                cache_prompt=previous is not None,
            )
            request_sha = sha256_bytes(canonical_json_bytes(request))
            response = _completion_request(client, request)
            observation = validate_completion_response(
                response,
                expected_input_tokens=len(prompt_tokens),
                expected_n_predict=current.n_predict,
                expected_slot_id=slot_id,
            )
            turn = TurnTranscript(
                lineage_id=current.lineage_id,
                turn=current.turn,
                timestamp=current.timestamp,
                input_length=current.input_length,
                n_predict=current.n_predict,
                type=current.type,
                block_ids=current.block_ids,
                prompt_tokens=prompt_tokens,
                reference_completion_tokens=observation.tokens,
                prompt_sha256=sha256_bytes(canonical_json_bytes(list(prompt_tokens))),
                reference_completion_sha256=sha256_bytes(
                    canonical_json_bytes(list(observation.tokens))
                ),
                request_sha256=request_sha,
                response_sha256=observation.response_sha256,
                slot_id=observation.slot_id,
                tokens_evaluated=observation.tokens_evaluated,
                tokens_predicted=observation.tokens_predicted,
            )
            executed_by_key[(current.lineage_id, current.turn)] = turn
            previous = turn
            request_hashes.append(request_sha)
            response_hashes.append(observation.response_sha256)

    process_start_after: Optional[dict[str, Any]] = None
    if calibration.real_model:
        assert authority.process_identity is not None
        process_start_after = capture_process_start_identity_snapshot(
            authority.process_identity, "reference pass after"
        )
        if process_start_after != process_start_before:
            raise ServerProcessIdentityError(
                "server process start identity changed during reference pass"
            )

    # Keep the frozen artifact in parent event/arrival order even though the
    # native reference requests must execute session-contiguously.
    frozen_turns = [
        executed_by_key[(current.lineage_id, current.turn)]
        for current in frozen.turns
    ]
    _check_cross_session_lcp(frozen_turns)
    is_real = (
        calibration.real_model
        and authority.source == "server_props_and_slots"
        and bool(authority.process_identity)
        and _HEX64.fullmatch(calibration.model_sha256 or "") is not None
        and _HEX64.fullmatch(calibration.binary_sha256 or "") is not None
        and authority.process_identity.get("model_sha256") == calibration.model_sha256
        and authority.process_identity.get("binary_sha256") == calibration.binary_sha256
    )
    parent = {
        "schema_version": frozen.manifest["schema_version"],
        "manifest_sha256": frozen.parent_manifest_sha256,
        "event_stream_sha256": frozen.parent_event_stream_sha256,
        "manifest_path_at_materialization": frozen.manifest_path,
        "events_path_at_materialization": frozen.events_path,
        "trace_key": frozen.manifest["trace_key"],
        "trace_repo_head": frozen.trace_repo_head,
        "trace_file": frozen.trace_file,
        "trace_file_sha256": frozen.trace_file_sha256,
        "slice_class": frozen.manifest["slice_class"],
    }
    model = {
        "model_sha256": calibration.model_sha256,
        "binary_sha256": calibration.binary_sha256,
        "identity_status": "REAL" if is_real else "UNVERIFIED",
        "server_process_identity": authority.process_identity,
    }
    reference_pass = {
        "name": REFERENCE_PASS_NAME,
        "request_count": len(frozen_turns),
        "completed": True,
        "request_sha256": sha256_bytes(canonical_json_bytes(request_hashes)),
        "response_sha256": sha256_bytes(canonical_json_bytes(response_hashes)),
        "generation_settings": dict(REFERENCE_GENERATION_SETTINGS),
        "execution_order": "session_contiguous",
        "artifact_order": "parent_event_order",
        "reference_completion_role": REFERENCE_COMPLETION_ROLE,
        "formal_correctness_oracle": FORMAL_CORRECTNESS_ORACLE,
        "reuse_contract": "expected_full_block_lcp",
        "block_lcp_observations": block_lcp_observations,
        "fixed_request_fields": {
            "cache_prompt": True,
            "cold_start_cache_prompt": False,
            "continuation_cache_prompt": True,
            "return_tokens": True,
            "ignore_eos": True,
            "stream": False,
            "stop": "absent",
            "prompt_kind": "direct_token_ids",
        },
        "process_start_identity_before": process_start_before,
        "process_start_identity_after": process_start_after,
    }
    transcript = TokenTranscript(
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        materialize_mode=MATERIALIZE_MODE_REAL if is_real else MATERIALIZE_MODE_OPEN,
        materialization_status=(
            MATERIALIZATION_STATUS_REAL if is_real else MATERIALIZATION_STATUS_UNVERIFIED
        ),
        parent=parent,
        model=model,
        effective_n_ctx_authority=authority.to_dict(),
        token_calibration=calibration.to_dict(),
        materializer=materializer.identity(),
        reference_pass=reference_pass,
        turns=frozen_turns,
        runtime_contract=runtime_contract if is_real else None,
    )
    return transcript.finalize()


def detect_policy_drift(
    transcript: TokenTranscript,
    *,
    lineage_id: int,
    turn: int,
    prompt_tokens: Optional[Sequence[int]] = None,
    completion_tokens: Optional[Sequence[int]] = None,
) -> Optional[str]:
    matches = [
        item
        for item in transcript.turns
        if item.lineage_id == lineage_id and item.turn == turn
    ]
    if len(matches) != 1:
        raise TranscriptDriftError(f"transcript turn not found: lineage={lineage_id} turn={turn}")
    expected = matches[0]
    if prompt_tokens is not None and tuple(prompt_tokens) != expected.prompt_tokens:
        return "prompt_drift"
    if completion_tokens is not None and tuple(completion_tokens) != expected.reference_completion_tokens:
        return "completion_drift"
    return None


def validate_policy_completion(
    transcript: TokenTranscript,
    *,
    lineage_id: int,
    turn: int,
    completion_tokens: Sequence[int],
) -> None:
    drift = detect_policy_drift(
        transcript,
        lineage_id=lineage_id,
        turn=turn,
        completion_tokens=completion_tokens,
    )
    if drift is not None:
        raise TranscriptDriftError(
            f"policy completion differs from reference transcript: {drift} "
            f"lineage={lineage_id} turn={turn}"
        )


def validate_policy_turn(
    transcript: TokenTranscript,
    *,
    lineage_id: int,
    turn: int,
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
) -> None:
    drift = detect_policy_drift(
        transcript,
        lineage_id=lineage_id,
        turn=turn,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    if drift is not None:
        raise TranscriptDriftError(
            f"policy turn differs from reference transcript: {drift} "
            f"lineage={lineage_id} turn={turn}"
        )


def build_dry_run_plan(frozen: FrozenSlice) -> dict[str, Any]:
    """Return an explicit non-formal plan without invoking a model."""
    return {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "materialize_mode": MATERIALIZE_MODE_OPEN,
        "materialization_status": MATERIALIZATION_STATUS_UNVERIFIED,
        "reason": "real tokenizer calibration and Resident reference pass were not run",
        "parent_manifest_sha256": frozen.parent_manifest_sha256,
        "parent_event_stream_sha256": frozen.parent_event_stream_sha256,
        "session_count": len(frozen.session_ids),
        "turn_count": len(frozen.turns),
        "formal_replay": False,
    }


def _write_json(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(value) + b"\n")


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace_compiler.runtime_materialize")
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run", help="validate parent and emit an OPEN plan")
    dry.add_argument("--manifest", required=True)
    dry.add_argument("--events")
    dry.add_argument("--out")
    run = sub.add_parser("materialize", help="run one Resident reference pass")
    run.add_argument("--manifest", required=True)
    run.add_argument("--events")
    run.add_argument("--out", required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--slot", type=int, default=0)
    run.add_argument("--model-sha")
    run.add_argument("--binary-sha")
    run.add_argument("--model-path")
    run.add_argument("--binary-path")
    run.add_argument("--server-pid", type=int)
    run.add_argument("--real-model", action="store_true")
    run.add_argument("--timeout", type=float, default=30.0)
    run.add_argument("--runtime-contract",
                     help="JSON file containing the exact runtime contract for REAL materialization")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _cli_parser().parse_args(argv)
    frozen = load_frozen_slice(args.manifest, args.events)
    if args.command == "dry-run":
        plan = build_dry_run_plan(frozen)
        if args.out:
            _write_json(args.out, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    client = LlamaServerClient(args.base_url, args.timeout)
    runtime_contract = None
    if args.runtime_contract:
        runtime_contract_document = _strict_json_object(
            Path(args.runtime_contract).read_bytes(), args.runtime_contract)
        runtime_contract = _strict_runtime_contract(
            runtime_contract_document, "runtime_contract")
    elif args.real_model:
        raise DirectTokenContractError(
            "real materialization requires --runtime-contract"
        )
    if args.real_model:
        if not args.model_path or not args.binary_path or args.server_pid is None:
            raise ServerProcessIdentityError(
                "real qualification requires --model-path, --binary-path, and --server-pid"
            )
        model_sha = sha256_file(args.model_path)
        binary_sha = sha256_file(args.binary_path)
        if args.model_sha is not None and args.model_sha != model_sha:
            raise ServerProcessIdentityError("--model-sha does not match model-path")
        if args.binary_sha is not None and args.binary_sha != binary_sha:
            raise ServerProcessIdentityError("--binary-sha does not match binary-path")
    else:
        model_sha = args.model_sha or (sha256_file(args.model_path) if args.model_path else None)
        binary_sha = args.binary_sha or (sha256_file(args.binary_path) if args.binary_path else None)

    # First obtain a provisional safe token from the live tokenizer.  Do not
    # mark this calibration as model-bound yet: the HTTP endpoint has not been
    # tied to --server-pid / model / binary identity at this point.
    probe_calibration = calibrate_tokenizer(client, real_model=False)

    authority = preflight_direct_token_contract(
        client,
        probe_token_id=probe_calibration.safe_alphabet[0],
        slot_id=args.slot,
    )
    if args.real_model:
        identity = verify_server_process_identity(
            authority,
            base_url=args.base_url,
            server_pid=args.server_pid,
            model_path=args.model_path,
            binary_path=args.binary_path,
            model_sha256=model_sha,
            binary_sha256=binary_sha,
            runtime_contract=runtime_contract,
        )
        authority = authority.bind_process_identity(identity)

    # Re-run the fixed calibration after process binding so the calibration
    # recorded in a REAL transcript is itself model/binary bound.
    calibration = calibrate_tokenizer(
        client,
        model_sha256=model_sha,
        binary_sha256=binary_sha,
        real_model=args.real_model,
    )
    if calibration.safe_alphabet != probe_calibration.safe_alphabet:
        raise CalibrationError(
            "tokenizer calibration changed across direct-token preflight/process binding"
        )
    transcript = materialize_reference_transcript(
        frozen,
        calibration,
        client,
        authority=authority,
        slot_id=args.slot,
        runtime_contract=runtime_contract,
    )
    transcript.write(args.out)
    print(json.dumps({
        "out": str(args.out),
        "schema_version": transcript.schema_version,
        "materialize_mode": transcript.materialize_mode,
        "materialization_status": transcript.materialization_status,
        "transcript_sha256": transcript.transcript_sha256,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
