import httpx
import logging
import json
from contextvars import ContextVar
from typing import Tuple, Dict, Any, Callable, Optional

from op_autoresearch.config import get_env_var
from .interface import WorkerInterface, empty_profile_result
from .eval_config import resolve_eval_timeout, resolve_reference_timeout
from op_autoresearch.cli.service.worker_config import worker_timing

logger = logging.getLogger(__name__)


# Connect / write / pool phase budgets and transient-retry count are read
# from the standard ``OP_AUTORESEARCH_WORKER_*`` env var layer (see
# ``op_autoresearch.config.get_env_var``). Operators can tune for
# flaky tunnels via ``export OP_AUTORESEARCH_WORKER_CONNECT_TIMEOUT_S=10``
# etc. without touching code. Defaults: connect short (5s) so dead ssh
# -L tunnels surface as ConnectError quickly; write/pool moderate;
# read timeout always comes from the per-call ``timeout`` arg because
# verify legitimately runs minutes on heavy DSLs.
def _connect_timeout_s() -> float:
    return float(get_env_var("WORKER_CONNECT_TIMEOUT_S", "5.0"))


def _write_timeout_s() -> float:
    return float(get_env_var("WORKER_WRITE_TIMEOUT_S", "30.0"))


def _pool_timeout_s() -> float:
    return float(get_env_var("WORKER_POOL_TIMEOUT_S", "5.0"))


def _transient_retry_attempts() -> int:
    """Max attempts (incl. first) for the ConnectError retry path. Only
    consulted when ``on_transient_failure`` is wired. Default 2 = one
    retry after invoking the callback. Set to 1 for fail-fast."""
    return max(1, int(get_env_var("WORKER_TRANSIENT_ATTEMPTS", "2")))


def _http_timeout(read_seconds: float) -> httpx.Timeout:
    """httpx.Timeout with connect/read/write/pool split so a hung daemon
    or dead tunnel can't make a single call swallow the full read budget."""
    return httpx.Timeout(
        connect=_connect_timeout_s(),
        read=read_seconds,
        write=_write_timeout_s(),
        pool=_pool_timeout_s(),
    )


class RemoteWorker(WorkerInterface):
    """
    Remote implementation of WorkerInterface.
    Delegates verification tasks to a remote VerificationService via HTTP.

    RemoteWorker manages the device pool of a remote server via HTTP API:
    -acquire_device(): request to assign device to remote server
    -release_device(): return device to remote server
    -Verify()/profile(): Send tasks to remote server execution

    ``on_transient_failure``: optional callback the worker invokes once
    after a ConnectError on long-running calls (verify/profile). Caller
    wires it to a local tunnel rebuild (typical: WA bridge's
    ``_make_reconnect_callback``) so a dead ssh -L tunnel auto-heals
    between attempts. The retry is single-shot — persistent failures
    still bubble up to the caller.
    """
    def __init__(self, worker_url: str,
                 on_transient_failure: Optional[Callable[[], None]] = None):
        self.worker_url = worker_url.rstrip('/')
        self.on_transient_failure = on_transient_failure

        # Lease state is coroutine-local. Concurrent jobs can share the same
        # op/task name without attaching or clearing each other's lease token.
        self._active_lease = ContextVar(
            f"remote_worker_lease_{id(self)}", default=None)

    def _attach_active_lease(self, data: Dict[str, Any], task_id: str
                             ) -> Dict[str, Any]:
        active = self._active_lease.get()
        if not active or active[0] != task_id:
            return data
        payload = dict(data)
        payload["device_id"] = str(active[1])
        payload["lease_id"] = str(active[2])
        return payload

    async def acquire_device(self, task_id: str = "unknown",
                             timeout: Optional[float] = None) -> Tuple[int, int]:
        """Fetches a usable device from a remote worker.

        ``timeout`` is device**Waiting budget**: pass to the service for queue waiting cap, yourself
        . http_read_margin saves more, so* the service gives up * and returns 503, instead of
        First read-timeout, leave a service that's still in line, waiter, just released.
        device to an expired clit. connect still by ``_connect_timeout_s``
        Founded (default 5s), tunnel breach Error → reconnect retry."""
        timing = worker_timing()
        wait_budget = float(timeout) if timeout is not None else timing.acquire_timeout
        read_timeout = wait_budget + timing.http_read_margin
        url = f"{self.worker_url}/api/v1/acquire_device"
        try:
            result = await self._post_with_reconnect(
                url, files=None,
                data={"task_id": task_id, "timeout": str(wait_budget)},
                read_timeout=read_timeout, task_id=task_id,
            )
            device_id = result.get("device_id")
            lease_id = result.get("lease_id")
            if not isinstance(device_id, int) or not isinstance(lease_id, int):
                raise RuntimeError(
                    f"worker returned invalid lease token: "
                    f"device_id={device_id!r}, lease_id={lease_id!r}")
            self._active_lease.set((task_id, device_id, lease_id))
            logger.info(f"[{task_id}] Acquired remote device {device_id} (lease {lease_id})")
            return device_id, lease_id
        except Exception as e:
            logger.error(f"[{task_id}] Failed to acquire remote device: {e}")
            raise RuntimeError(f"Failed to acquire remote device: {e}")

    async def release_device(self, device_id: int, lease_id: int,
                             task_id: str = "unknown"):
        """device was returned to the remote worker (with leave_id to block late release of the replaced lease)."""
        url = f"{self.worker_url}/api/v1/release_device"
        try:
            await self._post_with_reconnect(
                url, files=None,
                data={"task_id": task_id, "device_id": device_id, "lease_id": lease_id},
                read_timeout=worker_timing().release_timeout, task_id=task_id,
            )
            logger.info(f"[{task_id}] Released remote device {device_id}")
        except Exception as e:
            logger.error(f"[{task_id}] Failed to release remote device: {e}")
        finally:
            active = self._active_lease.get()
            if active == (task_id, device_id, lease_id):
                self._active_lease.set(None)

    async def get_doc(self, doc_name: str) -> str:
        """Pulls the contents of the document from a remote worker (GET, reconnect package)."""
        url = f"{self.worker_url}/api/v1/docs/{doc_name}"
        try:
            result = await self._get_with_reconnect(
                url, read_timeout=worker_timing().doc_timeout,
                task_id=f"doc:{doc_name}",
            )
            return result.get("content", "") if isinstance(result, dict) else ""
        except httpx.HTTPStatusError as e:
            logger.warning(
                "Remote worker returned %s for doc '%s': %s",
                e.response.status_code, doc_name, e.response.text,
            )
            return ""
        except Exception as e:
            logger.warning(
                "Failed to fetch remote doc '%s' from %s: %s",
                doc_name, self.worker_url, e,
            )
            return ""

    async def _post_with_reconnect(self, url: str, files, data,
                                   read_timeout: float, task_id: str):
        """POST helper: up to ``_transient_retry_attempts()`` attempts;
        on each ConnectError invoke ``on_transient_failure`` (if set) and
        retry. Other HTTP errors / read timeouts bubble up. Used by
        verify/profile AND acquire/release so a dead tunnel doesn't bypass
        the reconnect path."""
        attempts = (_transient_retry_attempts()
                    if self.on_transient_failure is not None else 1)
        last_exc = None
        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=_http_timeout(read_timeout)) as client:
                    response = await client.post(url, files=files, data=data)
                    response.raise_for_status()
                    return response.json()
            except httpx.ConnectError as e:
                last_exc = e
                if attempt + 1 < attempts:
                    logger.warning(
                        f"[{task_id}] Connection worker {self.worker_url} Failed "
                        f"(No.) {attempt + 1}/{attempts} Number of calls "
                        f"on_transient_failure Try again after"
                    )
                    try:
                        self.on_transient_failure()
                    except Exception as cb_err:
                        logger.error(
                            f"[{task_id}] on_transient_failure Drop the anomaly:{cb_err}"
                        )
                    continue
                raise
        raise last_exc  # unreachable; satisfies the type checker

    async def _get_with_reconnect(self, url: str, *, read_timeout: float,
                                  task_id: str):
        """GET twin of ``_post_with_reconnect`` — same retry/reconnect
        policy. Used by ``get_doc``."""
        attempts = (_transient_retry_attempts()
                    if self.on_transient_failure is not None else 1)
        last_exc = None
        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=_http_timeout(read_timeout)) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    return response.json()
            except httpx.ConnectError as e:
                last_exc = e
                if attempt + 1 < attempts:
                    logger.warning(
                        f"[{task_id}] GET {url} ConnectError "
                        f"(No.) {attempt + 1}/{attempts} Number of calls "
                        f"on_transient_failure Try again after"
                    )
                    try:
                        self.on_transient_failure()
                    except Exception as cb_err:
                        logger.error(
                            f"[{task_id}] on_transient_failure Drop the anomaly:{cb_err}"
                        )
                    continue
                raise
        raise last_exc

    async def verify(self, package_data: bytes, task_id: str, op_name: str,
                     timeout: Optional[int] = None
                     ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Send verification task to remote worker.

        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, log, artifacts)
        """
        timeout = resolve_eval_timeout(timeout)
        verify_url = f"{self.worker_url}/api/v1/verify"

        try:
            files = {'package': ('package.tar', package_data, 'application/x-tar')}
            data = self._attach_active_lease({
                'task_id': task_id,
                'op_name': op_name,
                'timeout': str(timeout)
            }, task_id)
            logger.info(f"[{task_id}] Sending verification request to {verify_url}")

            result = await self._post_with_reconnect(
                verify_url, files=files, data=data,
                read_timeout=timeout + worker_timing().http_read_margin,
                task_id=task_id,
            )
            success = result.get('success', False)
            log = result.get('log', '')
            artifacts = result.get('artifacts', {})

            if artifacts:
                logger.info(f"[{task_id}] Received {len(artifacts)} artifact files from remote worker")

            return success, log, artifacts

        except httpx.RequestError as e:
            error_msg = f"Network error communicating with worker at {self.worker_url}: {e}. Please check if the worker service is running and accessible."
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, {}
        except httpx.HTTPStatusError as e:
            error_msg = f"Worker returned error status: {e.response.status_code} - {e.response.text}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, {}
        except Exception as e:
            error_msg = f"Remote verification failed: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, {}

    async def profile(self, package_data: bytes, task_id: str, op_name: str, profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send profiling task to remote worker.

        Returns:
            Dict[str, Any]: Include gen_time, base_time, Speedup, phrases
        """
        profile_url = f"{self.worker_url}/api/v1/profile"
        timeout = resolve_eval_timeout(profile_settings.get('timeout'))
        try:
            files = {'package': ('package.tar', package_data, 'application/x-tar')}
            data = self._attach_active_lease({
                'task_id': task_id,
                'op_name': op_name,
                'profile_settings': json.dumps(profile_settings)
            }, task_id)
            logger.info(f"[{task_id}] Sending profiling request to {profile_url}")

            result = await self._post_with_reconnect(
                profile_url, files=files, data=data,
                # LocalWorker profiles base then generation serially, and
                # ``timeout`` is the per-script wall cap.  The HTTP request
                # must cover both caps; otherwise the client can abandon a
                # live server-side generation profile after a slow base run.
                read_timeout=2 * timeout + worker_timing().http_read_margin,
                task_id=task_id,
            )
            artifacts = result.get('artifacts', {})
            if artifacts:
                logger.info(f"[{task_id}] Received {len(artifacts)} artifact files from remote worker")

            return result

        except Exception as e:
            logger.error(f"[{task_id}] Remote profiling failed: {e}")
            return empty_profile_result(error=str(e))

    async def profile_single_task(self, package_data: bytes, task_id: str, op_name: str,
                                   profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send single task profiling request to remote worker.

        A separate measure of the performance of a certain section of the code does not allow comparison.

        Returns:
            Dict[str, Any]: Paragraph containing time_us, access, log
        """
        profile_url = f"{self.worker_url}/api/v1/profile_single_task"
        timeout = resolve_eval_timeout(profile_settings.get('timeout'))
        try:
            files = {'package': ('package.tar', package_data, 'application/x-tar')}
            data = self._attach_active_lease({
                'task_id': task_id,
                'op_name': op_name,
                'profile_settings': json.dumps(profile_settings)
            }, task_id)
            logger.info(f"[{task_id}] Sending profile_single_task request to {profile_url}")

            result = await self._post_with_reconnect(
                profile_url, files=files, data=data,
                read_timeout=timeout + worker_timing().http_read_margin,
                task_id=task_id,
            )
            return result

        except httpx.RequestError as e:
            error_msg = f"Network error communicating with worker at {self.worker_url}: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return {'time_us': None, 'success': False, 'log': error_msg}
        except httpx.HTTPStatusError as e:
            error_msg = f"Worker returned error status: {e.response.status_code} - {e.response.text}"
            logger.error(f"[{task_id}] {error_msg}")
            return {'time_us': None, 'success': False, 'log': error_msg}
        except Exception as e:
            error_msg = f"Remote profile_single_task failed: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return {'time_us': None, 'success': False, 'log': error_msg}

    async def generate_reference(self, package_data: bytes, task_id: str,
                                 op_name: str, timeout: Optional[int] = None
                                 ) -> Tuple[bool, str, bytes]:
        """
        Send reference generation task to remote worker.

        For the CUDA-to-Asend conversion scenario: execute the Triton-CUDA code on a remote GPU Worker.
        Generates reference data (.pt file) and returns its binary content.

        Args:
            Package_data: Verify package data (TAR bytes)
            task_id: Task ID
            Op_name: operator name
            Timeout: Timeout

        Returns:
            Tuple[bool, str, bytes]: (success, log, reference_data_bytes)
        """
        timeout = resolve_reference_timeout(timeout)
        import base64

        generate_ref_url = f"{self.worker_url}/api/v1/generate_reference"

        try:
            files = {'package': ('package.tar', package_data, 'application/x-tar')}
            data = self._attach_active_lease({
                'task_id': task_id,
                'op_name': op_name,
                'timeout': str(timeout),
            }, task_id)
            logger.info(f"[{task_id}] Sending generate_reference request to {generate_ref_url}")
            # Walk uniform packaging - Previously n nudised httpx. AsyncClint
            # Breaktime does not trigger on_transient_fair, cross
            # Back reference data generation will be unresponsive.
            result = await self._post_with_reconnect(
                generate_ref_url, files=files, data=data,
                read_timeout=timeout + worker_timing().http_read_margin,
                task_id=task_id,
            )
            success = result.get('success', False)
            log = result.get('log', '')

            if success:
                # transfer_data encoded in base64
                ref_data_b64 = result.get('reference_data', '')
                if ref_data_b64:
                    ref_bytes = base64.b64decode(ref_data_b64)
                    logger.info(f"[{task_id}] Received reference data: {len(ref_bytes)} bytes")
                    return True, log, ref_bytes
                return False, f"No reference data in response:\n{log}", b''
            return False, log, b''

        except httpx.RequestError as e:
            error_msg = f"Network error communicating with worker at {self.worker_url}: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, b''
        except httpx.HTTPStatusError as e:
            error_msg = f"Worker returned error status: {e.response.status_code} - {e.response.text}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, b''
        except Exception as e:
            error_msg = f"Remote generate_reference failed: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, b''
