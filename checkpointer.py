"""
checkpointer.py — minimal Redis-backed LangGraph checkpoint saver.

Why hand-rolled instead of the official `langgraph-checkpoint-redis` package: that package's
RedisSaver/ShallowRedisSaver require RedisJSON + RediSearch (Redis Stack, or Redis >=8 with
those modules bundled). Nothing else in this codebase assumes those modules are present —
`memory.py` uses plain GET/SET, and the production REDIS_URL's actual module support is
unverified. A plain Redis instance would fail at index-creation time with that package.

This saver ports langgraph.checkpoint.memory.InMemorySaver's storage model (read its source
for the reference this is a direct translation of) onto Redis hashes with a TTL, so a paused
workflow survives across separate HTTP requests *and* process restarts (e.g. a deploy) —
which a purely in-memory saver would not.

Scope deliberately narrow: only the async methods (`aget_tuple`, `aput`, `aput_writes`) are
implemented, since this service only ever calls `graph.ainvoke(...)`. The sync counterparts,
`alist`/`list` (checkpoint history), and delete/copy/prune are intentionally left as the base
class's NotImplementedError — this graph has exactly one interrupt, no subgraphs, no branching,
and no need to time-travel across checkpoints, so that surface is genuinely unused.
"""
import base64
import json
import logging

import redis.asyncio as redis
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from config import settings

logger = logging.getLogger(__name__)

_redis: redis.Redis | None = None


def _client() -> redis.Redis:
    global _redis
    if _redis is None:
        # decode_responses=False here (unlike memory.py) — values are base64-wrapped
        # serde bytes, and we want them back as bytes/str exactly as written, not
        # auto-decoded in a way that could mangle binary payloads.
        _redis = redis.from_url(settings.redis_url, decode_responses=False)
    return _redis


async def acquire_resume_lock(workflow_id: str, ttl_seconds: int) -> bool:
    """Short-lived NX lock guarding a resume (approve/reject) call — the FastAPI route
    handler wraps ainvoke(Command(resume=...)) in this, not the graph itself (a node
    re-executes its whole body on resume, so locking inside a node would be wrong —
    see graphs/fee_reminder_workflow.py's module docstring)."""
    return bool(await _client().set(f"aiflow:{workflow_id}:resume-lock", "1", nx=True, ex=ttl_seconds))


async def release_resume_lock(workflow_id: str) -> None:
    await _client().delete(f"aiflow:{workflow_id}:resume-lock")


async def close() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


class RedisCheckpointSaver(BaseCheckpointSaver):
    """One saver instance is shared process-wide; all state lives in Redis, not on `self`."""

    def __init__(self, ttl_seconds: int, serde: SerializerProtocol | None = None):
        super().__init__(serde=serde)
        self.ttl_seconds = ttl_seconds

    def _checkpoints_key(self, thread_id: str) -> str:
        return f"aiflow:{thread_id}:checkpoints"

    def _blobs_key(self, thread_id: str) -> str:
        return f"aiflow:{thread_id}:blobs"

    def _writes_key(self, thread_id: str, checkpoint_id: str) -> str:
        return f"aiflow:{thread_id}:writes:{checkpoint_id}"

    async def _touch_ttl(self, thread_id: str, *extra_keys: str) -> None:
        client = _client()
        pipe = client.pipeline()
        for k in (self._checkpoints_key(thread_id), self._blobs_key(thread_id), *extra_keys):
            pipe.expire(k, self.ttl_seconds)
        await pipe.execute()

    async def _load_blobs(self, thread_id: str, channel_versions: ChannelVersions) -> dict:
        if not channel_versions:
            return {}
        client = _client()
        channels = list(channel_versions.keys())
        fields = [f"{ch}:{channel_versions[ch]}" for ch in channels]
        raw_values = await client.hmget(self._blobs_key(thread_id), fields)
        result: dict = {}
        for channel, raw in zip(channels, raw_values):
            if raw is None:
                continue
            v_type, v_b64 = json.loads(raw)
            if v_type == "empty":
                continue
            result[channel] = self.serde.loads_typed((v_type, _unb64(v_b64)))
        return result

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        client = _client()

        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id is None:
            all_ids = await client.hkeys(self._checkpoints_key(thread_id))
            if not all_ids:
                return None
            checkpoint_id = max(i.decode("utf-8") for i in all_ids)

        raw = await client.hget(self._checkpoints_key(thread_id), checkpoint_id)
        if raw is None:
            return None

        checkpoint_type, checkpoint_b64, metadata_type, metadata_b64, parent_id = json.loads(raw)
        checkpoint_: Checkpoint = self.serde.loads_typed((checkpoint_type, _unb64(checkpoint_b64)))
        metadata: CheckpointMetadata = self.serde.loads_typed((metadata_type, _unb64(metadata_b64)))
        channel_values = await self._load_blobs(thread_id, checkpoint_["channel_versions"])

        raw_writes = await client.hgetall(self._writes_key(thread_id, checkpoint_id))
        pending_writes = []
        for value in raw_writes.values():
            task_id, channel, v_type, v_b64, _task_path = json.loads(value)
            pending_writes.append((task_id, channel, self.serde.loads_typed((v_type, _unb64(v_b64)))))

        return CheckpointTuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_ns": "", "checkpoint_id": checkpoint_id}},
            checkpoint={**checkpoint_, "channel_values": channel_values},
            metadata=metadata,
            pending_writes=pending_writes,
            parent_config=(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": "", "checkpoint_id": parent_id}}
                if parent_id else None
            ),
        )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        client = _client()

        c = dict(checkpoint)
        values = c.pop("channel_values")

        pipe = client.pipeline()
        for channel, version in new_versions.items():
            if channel in values:
                v_type, v_bytes = self.serde.dumps_typed(values[channel])
                blob = json.dumps([v_type, _b64(v_bytes)])
            else:
                blob = json.dumps(["empty", ""])
            pipe.hset(self._blobs_key(thread_id), f"{channel}:{version}", blob)

        checkpoint_type, checkpoint_bytes = self.serde.dumps_typed(c)
        metadata_type, metadata_bytes = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        parent_id = config["configurable"].get("checkpoint_id")
        stored = json.dumps([
            checkpoint_type, _b64(checkpoint_bytes),
            metadata_type, _b64(metadata_bytes),
            parent_id,
        ])
        pipe.hset(self._checkpoints_key(thread_id), checkpoint["id"], stored)
        await pipe.execute()
        await self._touch_ttl(thread_id)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes,
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        client = _client()
        key = self._writes_key(thread_id, checkpoint_id)

        existing_raw = await client.hkeys(key)
        existing = {f.decode("utf-8") for f in existing_raw}

        pipe = client.pipeline()
        wrote_any = False
        for idx, (channel, value) in enumerate(writes):
            # Special channels (error/scheduled/interrupt/resume) get fixed negative
            # indices and are always overwritten; ordinary channel writes are keyed by
            # their position and written only once per (task_id, idx) — mirrors
            # InMemorySaver.put_writes exactly, see its docstring for why.
            inner_idx = WRITES_IDX_MAP.get(channel, idx)
            field = f"{task_id}:{inner_idx}"
            if inner_idx >= 0 and field in existing:
                continue
            v_type, v_bytes = self.serde.dumps_typed(value)
            entry = json.dumps([task_id, channel, v_type, _b64(v_bytes), task_path])
            pipe.hset(key, field, entry)
            wrote_any = True

        if wrote_any:
            await pipe.execute()
            await self._touch_ttl(thread_id, key)
