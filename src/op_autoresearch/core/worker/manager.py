from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set
import asyncio
import logging
from .interface import WorkerInterface
from op_autoresearch.config import get_env_var
from op_autoresearch.cli.service.worker_config import worker_timing

logger = logging.getLogger(__name__)

@dataclass
class WorkerInfo:
    worker: WorkerInterface
    backend: str
    arch: str
    tags: Set[str] = field(default_factory=set)
    capacity: int = 1
    load: int = 0

class WorkerManager:
    """Register, discover, and load-balance workers.

    This manager tracks routing load but does not lock hardware resources.
    Each worker's ``DevicePool`` provides resource exclusion.
    """
    def __init__(self):
        self._workers: List[WorkerInfo] = []
        self._lock = asyncio.Lock()

    async def register(self, worker: WorkerInterface, backend: str, arch: str, tags: Set[str] = None, capacity: int = 1):
        """Register a worker.

        Args:
            worker: Worker instance.
            backend: Backend type, such as ``cuda``, ``ascend``, or ``cpu``.
            arch: Hardware architecture.
            tags: Optional labels, such as ``local`` or ``fast``.
            capacity: Maximum concurrent load, usually the device count.
        """
        async with self._lock:
            info = WorkerInfo(
                worker=worker,
                backend=backend,
                arch=arch,
                tags=tags or set(),
                capacity=max(1, capacity)
            )
            self._workers.append(info)
            logger.info(f"Registered worker: backend={backend}, arch={arch}, capacity={capacity}")

    async def select(self, backend: str, arch: Optional[str] = None, tags: Set[str] = None) -> Optional[WorkerInterface]:
        """Select the least-loaded matching worker.

        Policy:
        1. Filter by backend, architecture, and tags.
        2. Choose the lowest ``load / capacity`` ratio.
        3. Increment its routing load counter.

        Returns:
            The selected worker, or ``None`` when no worker matches.
        """
        async with self._lock:
            candidates = []
            for info in self._workers:
                if info.backend != backend:
                    continue
                if arch and info.arch != arch:
                    continue
                if tags and not tags.issubset(info.tags):
                    continue
                candidates.append(info)

            if not candidates:
                return None

            # Select the lowest normalized load.
            best_info = min(candidates, key=lambda w: w.load / w.capacity)

            best_info.load += 1
            logger.debug(f"Selected worker {id(best_info.worker)} (load={best_info.load}/{best_info.capacity})")
            return best_info.worker

    async def reserve(self, worker: WorkerInterface) -> bool:
        """Increment load for one already-registered worker instance."""
        async with self._lock:
            for info in self._workers:
                if info.worker is worker:
                    info.load += 1
                    return True
            return False

    async def has_worker(self, backend: str, arch: Optional[str] = None, tags: Set[str] = None) -> bool:
        """Return whether a worker matches without changing its load."""
        async with self._lock:
            for info in self._workers:
                if info.backend != backend:
                    continue
                if arch and info.arch != arch:
                    continue
                if tags and not tags.issubset(info.tags):
                    continue
                return True
            return False

    async def list_matching(self, backend: str, arch: Optional[str] = None, tags: Set[str] = None) -> List[WorkerInterface]:
        """Return all matching workers without changing their loads."""
        async with self._lock:
            return [
                info.worker
                for info in self._workers
                if info.backend == backend
                and (not arch or info.arch == arch)
                and (not tags or tags.issubset(info.tags))
            ]

    async def release(self, worker: WorkerInterface) -> bool:
        """Decrement a worker's load after task completion or failure."""
        async with self._lock:
            for info in self._workers:
                if info.worker is worker:
                    info.load = max(0, info.load - 1)
                    logger.debug(f"Released worker {id(info.worker)} (load={info.load}/{info.capacity})")
                    return True
            logger.error(f"Release called for unknown worker id={id(worker)}")
            return False

    async def get_status(self) -> List[dict]:
        """Return state snapshots for all registered workers."""
        async with self._lock:
            return [
                {
                    "backend": w.backend,
                    "arch": w.arch,
                    "load": w.load,
                    "capacity": w.capacity,
                    "tags": list(w.tags)
                }
                for w in self._workers
            ]

# Process-wide manager.
_GLOBAL_MANAGER = WorkerManager()

def get_worker_manager() -> WorkerManager:
    return _GLOBAL_MANAGER

async def register_local_worker(device_ids: List[int], backend: str, arch: str,
                                tags: Set[str] = None) -> WorkerInterface:
    """Create and register a ``LocalWorker`` with the global manager.

    Args:
        device_ids: Device IDs, for example ``[0]`` or ``[0, 1, 2, 3]``.
        backend: Backend type.
        arch: Hardware architecture.
        tags: Optional labels.

    Example:
        # One device
        await register_local_worker([0], backend="cuda", arch="a100")

        # Multiple devices
        await register_local_worker([0, 1, 2, 3], backend="cuda", arch="a100")
    """
    from .local_worker import LocalWorker
    from ..async_pool.device_pool import DevicePool

    device_pool = DevicePool(device_ids)
    local_worker = LocalWorker(device_pool, backend=backend)
    await _GLOBAL_MANAGER.register(
        local_worker,
        backend=backend,
        arch=arch,
        tags=tags,
        capacity=len(device_ids)
    )
    logger.info(f"✅ Registered LocalWorker: backend={backend}, arch={arch}, devices={device_ids}")

    return local_worker

async def register_remote_worker(backend: str, arch: str, worker_url: Optional[str] = None, capacity: Optional[int] = None, tags: Set[str] = None,
                                 expected_device_ids: Optional[List[int]] = None,
                                 on_transient_failure: Optional[Callable[[], None]] = None) -> WorkerInterface:
    """Create and register a ``RemoteWorker`` with the global manager.

    When ``worker_url`` is omitted, read ``OP_AUTORESEARCH_WORKER_URL``.
    When ``capacity`` is omitted, derive it from the remote status endpoint.

    Args:
        backend: Backend type.
        arch: Hardware architecture.
        worker_url: Optional remote worker service URL.
        capacity: Optional concurrency capacity, usually the device count.
        tags: Optional labels.
        expected_device_ids: Device IDs expected from the remote status probe.
        on_transient_failure: Optional callback for transient connection errors.

    Example:
        # Read the URL from the environment and discover capacity.
        export OP_AUTORESEARCH_WORKER_URL=http://localhost:9001
        await register_remote_worker(backend="cuda", arch="a100")

        # Specify both URL and capacity.
        await register_remote_worker(
            backend="cuda",
            arch="a100",
            worker_url="http://localhost:9001",
            capacity=2
        )
    """
    import httpx
    from .remote_worker import RemoteWorker

    if worker_url is None:
        worker_url = get_env_var("WORKER_URL")
        if worker_url is None:
            raise ValueError(
                "worker_url was not provided and OP_AUTORESEARCH_WORKER_URL is not set.\n"
                "Provide worker_url or set the environment variable:\n"
                "  export OP_AUTORESEARCH_WORKER_URL=http://localhost:9001"
            )

    # Probe /status: derives capacity when not given, AND asserts the
    # daemon's reported devices intersect `expected_device_ids` so we
    # don't silently bind the task to the wrong daemon (e.g. someone
    # else's worker happens to be listening on the same local tunnel
    # port). Mismatch raises rather than warning — silent device drift
    # produces correct-looking but invalid benchmark numbers.
    remote_devices: List[int] = []
    if capacity is None or expected_device_ids:
        try:
            status_url = f"{worker_url.rstrip('/')}/api/v1/status"
            async with httpx.AsyncClient(
                timeout=worker_timing().status_timeout
            ) as client:
                response = await client.get(status_url)
                response.raise_for_status()
                status = response.json()
                devs = status.get("devices", [])
                if isinstance(devs, list):
                    remote_devices = [int(d) for d in devs]
        except Exception as e:
            if expected_device_ids:
                raise RuntimeError(
                    f"Unable to verify worker {worker_url}: /status probe failed ({e}). "
                    f"Expected device IDs: {expected_device_ids}."
                )
            logger.warning("Remote worker status probe failed: %s; using capacity=1", e)
        if expected_device_ids:
            if not remote_devices:
                raise RuntimeError(
                    f"Worker {worker_url} returned an empty device list; cannot verify "
                    f"the requested devices {sorted(expected_device_ids)}. Common causes "
                    "are an older daemon whose /status omits devices or an SSH tunnel "
                    "connected to a daemon that has not finished initialization."
                )
            if not (set(remote_devices) & set(expected_device_ids)):
                raise RuntimeError(
                    f"Worker {worker_url} device mismatch: the daemon reports "
                    f"devices={remote_devices}, but the task requests "
                    f"{sorted(expected_device_ids)}. Check task.yaml worker.urls and "
                    "confirm that the SSH tunnel reaches the intended daemon."
                )
        if capacity is None:
            capacity = len(remote_devices) if remote_devices else 1
            logger.info("Remote worker reports %s devices: %s", capacity, remote_devices)

    remote_worker = RemoteWorker(worker_url, on_transient_failure=on_transient_failure)
    await _GLOBAL_MANAGER.register(
        remote_worker,
        backend=backend,
        arch=arch,
        tags=tags,
        capacity=max(1, capacity)
    )
    logger.info(f"✅ Registered RemoteWorker: backend={backend}, arch={arch}, url={worker_url}, capacity={capacity}")

    return remote_worker

async def register_worker(
    backend: str,
    arch: str,
    *,
    device_ids: Optional[List[int]] = None,
    worker_url: Optional[str] = None,
    tags: Optional[Set[str]] = None,
    on_transient_failure: Optional[Callable[[], None]] = None,
) -> WorkerInterface:
    """Register a remote or local worker.

    Priority:
      1. Use a remote worker when ``worker_url`` or
         ``OP_AUTORESEARCH_WORKER_URL`` is set.
      2. Otherwise, use a local worker when ``device_ids`` is provided.
      3. Raise an actionable error when neither source is configured.

    On the remote path, ``device_ids`` is an expected-device set rather than
    a list of local devices. Registration fails when it does not intersect the
    daemon's reported devices, preventing an SSH tunnel from silently reaching
    the wrong daemon. ``on_transient_failure`` is passed to ``RemoteWorker`` so
    callers can rebuild the tunnel after connection failures.
    """
    resolved_worker_url = worker_url or get_env_var("WORKER_URL")
    if resolved_worker_url:
        return await register_remote_worker(
            backend=backend,
            arch=arch,
            worker_url=resolved_worker_url,
            tags=tags,
            expected_device_ids=device_ids,
            on_transient_failure=on_transient_failure,
        )

    if device_ids:
        return await register_local_worker(
            device_ids, backend=backend, arch=arch, tags=tags)

    raise RuntimeError(
        "No worker is available. Register a worker before running evolve:\n"
        "  Option 1: set a remote worker URL\n"
        "    export OP_AUTORESEARCH_WORKER_URL=http://<worker-host>:<port>\n"
        "  Option 2: call register_worker(..., device_ids=[0]) for local devices"
    )
