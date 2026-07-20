from __future__ import annotations

import hmac
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import quote, urlparse

from ..config import R2Config
from ..env import load_env
from .manifest import CurrentPointer, DataRelease, DatasetManifest, sha256_bytes, write_atomic
from .schemas import dataset_spec


class ConditionalWriteFailed(RuntimeError):
    pass


class ObjectNotFound(FileNotFoundError):
    pass


class R2PrivacyVerificationError(RuntimeError):
    """Raised before a write when private R2 visibility is not proven."""


class R2PrivacyVerificationUnavailable(R2PrivacyVerificationError):
    """Raised when the authoritative Cloudflare visibility API is unavailable."""


@dataclass(frozen=True)
class ObjectValue:
    data: bytes
    etag: str


class ObjectStore(Protocol):
    def get(self, key: str) -> ObjectValue:
        ...

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
    ) -> str:
        ...

    def list(self, prefix: str) -> tuple[str, ...]:
        ...


class LocalObjectStore:
    """Filesystem implementation used for local operation and CAS tests."""

    def __init__(self, root: str | Path):
        # Keep one canonical form.  ``_path`` resolves object keys, so retaining
        # a relative root makes ``list`` compare absolute descendants against a
        # relative base and raises ``ValueError`` in normal CLI configurations
        # such as ``local_cache_dir: data/cache``.
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key.lstrip("/")).resolve()
        root = self.root
        if path != root and root not in path.parents:
            raise ValueError(f"Object key escapes store root: {key}")
        return path

    def get(self, key: str) -> ObjectValue:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFound(key)
        data = path.read_bytes()
        return ObjectValue(data=data, etag=sha256_bytes(data))

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
    ) -> str:
        path = self._path(key)
        existing = path.read_bytes() if path.is_file() else None
        if if_none_match and existing is not None:
            raise ConditionalWriteFailed(f"Object already exists: {key}")
        if if_match is not None:
            actual = sha256_bytes(existing) if existing is not None else ""
            if actual != if_match.strip('"'):
                raise ConditionalWriteFailed(f"ETag changed for {key}")
        write_atomic(path, data)
        return sha256_bytes(data)

    def list(self, prefix: str) -> tuple[str, ...]:
        base = self._path(prefix)
        if base.is_file():
            return (prefix,)
        if not base.exists():
            return ()
        return tuple(
            str(path.relative_to(self.root)).replace(os.sep, "/")
            for path in sorted(base.rglob("*"))
            if path.is_file()
        )


_CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"
_PUBLIC_ACL_URIS = (
    "acs.amazonaws.com/groups/global/allusers",
    "acs.amazonaws.com/groups/global/authenticatedusers",
)


def _r2_endpoint_identity(endpoint_url: str) -> tuple[str, str]:
    parsed = urlparse(endpoint_url)
    host = (parsed.hostname or "").lower()
    suffix = ".r2.cloudflarestorage.com"
    if parsed.scheme != "https" or not host.endswith(suffix):
        raise R2PrivacyVerificationError(
            "R2 privacy verification requires the official HTTPS "
            "*.r2.cloudflarestorage.com S3 endpoint."
        )
    account_id = host.split(".", 1)[0]
    if len(account_id) != 32 or any(
        character not in "0123456789abcdef" for character in account_id
    ):
        raise R2PrivacyVerificationError(
            "R2 account identity cannot be derived from the S3 endpoint."
        )
    return account_id, host


def _parse_attestation_time(value: Any, field: str) -> datetime:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise R2PrivacyVerificationError(
            f"R2 privacy attestation has an invalid {field}."
        ) from exc
    if parsed.tzinfo is None:
        raise R2PrivacyVerificationError(
            f"R2 privacy attestation {field} must include a timezone."
        )
    return parsed.astimezone(timezone.utc)


def _policy_allows_public_access(policy: Any) -> bool:
    if isinstance(policy, str):
        try:
            policy = json.loads(policy)
        except (TypeError, ValueError):
            return True
    if not isinstance(policy, dict):
        return True
    statements = policy.get("Statement", ())
    if isinstance(statements, dict):
        statements = (statements,)
    if not isinstance(statements, (list, tuple)):
        return True
    for statement in statements:
        if not isinstance(statement, dict):
            return True
        if str(statement.get("Effect") or "").lower() != "allow":
            continue
        principal = statement.get("Principal")
        if principal is None:
            return True
        public = principal == "*"
        if isinstance(principal, dict):
            for value in principal.values():
                if value == "*":
                    public = True
                    break
                if isinstance(value, (list, tuple)):
                    if "*" in value or any(not isinstance(item, str) for item in value):
                        public = True
                        break
                elif not isinstance(value, str):
                    public = True
                    break
        elif not isinstance(principal, str):
            return True
        if public:
            return True
    return False


class R2PrivacyVerifier:
    """Fail-closed R2 visibility verifier used before the first object write.

    R2's S3-compatible API does not currently expose authoritative public
    bucket state. We still probe its ACL/policy/public-access operations first
    and reject any positive public finding. Final proof comes from Cloudflare's
    managed-domain and custom-domain REST endpoints, or from a short-lived
    hash-pinned attestation containing either those exact API checks or an
    operator-reviewed Cloudflare dashboard screenshot.
    """

    def __init__(
        self,
        config: R2Config,
        s3_client,
        endpoint_url: str,
        *,
        environ: Mapping[str, str] | None = None,
        http_session=None,
        now: Callable[[], datetime] | None = None,
    ):
        self.config = config
        self.s3_client = s3_client
        self.endpoint_url = endpoint_url
        self.environ = os.environ if environ is None else environ
        self.http_session = http_session
        self.now = now or (lambda: datetime.now(timezone.utc))

    def verify(self) -> dict[str, Any]:
        endpoint_account, endpoint_host = _r2_endpoint_identity(self.endpoint_url)
        s3_checks = self._inspect_s3_visibility()
        configured_account_id = str(
            self.environ.get(self.config.account_id_env, "")
        ).strip()
        api_token = str(self.environ.get(self.config.api_token_env, "")).strip()
        attestation_path = str(
            self.environ.get(self.config.privacy_attestation_path_env, "")
        ).strip()
        attestation_hash = str(
            self.environ.get(self.config.privacy_attestation_sha256_env, "")
        ).strip()

        if configured_account_id and not api_token:
            if not (attestation_path and attestation_hash):
                raise R2PrivacyVerificationError(
                    "R2 private-state verification requires "
                    f"{self.config.api_token_env} when "
                    f"{self.config.account_id_env} is set, or both "
                    "privacy-attestation environment values."
                )
        if configured_account_id and configured_account_id != endpoint_account:
            raise R2PrivacyVerificationError(
                "Cloudflare account ID does not match the configured R2 endpoint."
            )
        if api_token:
            # The official R2 S3 endpoint is account-scoped, so a separate
            # account-id secret is redundant.  Accept an explicit value only
            # as a consistency assertion and otherwise derive it from the
            # already validated HTTPS endpoint.
            account_id = configured_account_id or endpoint_account
            if account_id != endpoint_account:
                raise R2PrivacyVerificationError(
                    "Cloudflare account ID does not match the configured R2 endpoint."
                )
            try:
                result = self._verify_cloudflare_api(account_id, api_token)
            except R2PrivacyVerificationUnavailable:
                if not (attestation_path and attestation_hash):
                    raise
            else:
                return {**result, "s3_checks": s3_checks}

        if not (attestation_path and attestation_hash):
            raise R2PrivacyVerificationError(
                "R2 public-domain state cannot be proven through the S3 API. "
                f"Set {self.config.api_token_env} (the account ID is derived "
                "from the R2 endpoint), or provide both hash-pinned "
                "privacy-attestation values."
            )
        result = self._verify_attestation(
            Path(attestation_path),
            attestation_hash,
            endpoint_account=endpoint_account,
            endpoint_host=endpoint_host,
        )
        return {**result, "s3_checks": s3_checks}

    def _inspect_s3_visibility(self) -> dict[str, str]:
        try:
            self.s3_client.head_bucket(Bucket=self.config.bucket)
        except Exception as exc:
            code = _client_error_code(exc) or type(exc).__name__
            raise R2PrivacyVerificationError(
                f"R2 bucket identity check failed before publication ({code})."
            ) from None

        checks: dict[str, str] = {"head_bucket": "passed"}
        probes = (
            ("public_access_block", "get_public_access_block"),
            ("bucket_policy_status", "get_bucket_policy_status"),
            ("bucket_policy", "get_bucket_policy"),
            ("bucket_acl", "get_bucket_acl"),
        )
        for label, method_name in probes:
            method = getattr(self.s3_client, method_name, None)
            if not callable(method):
                checks[label] = "unsupported"
                continue
            try:
                response = method(Bucket=self.config.bucket)
            except Exception as exc:
                code = _client_error_code(exc) or type(exc).__name__
                checks[label] = f"unavailable:{code}"
                continue
            if not isinstance(response, dict):
                raise R2PrivacyVerificationError(
                    f"R2 S3 {label} returned an invalid response."
                )
            if label == "public_access_block":
                block = response.get("PublicAccessBlockConfiguration")
                protected = isinstance(block, dict) and all(
                    block.get(key) is True
                    for key in (
                        "BlockPublicAcls",
                        "IgnorePublicAcls",
                        "BlockPublicPolicy",
                        "RestrictPublicBuckets",
                    )
                )
                checks[label] = "protective" if protected else "inconclusive"
            elif label == "bucket_policy_status":
                status = response.get("PolicyStatus")
                if isinstance(status, dict) and status.get("IsPublic") is True:
                    raise R2PrivacyVerificationError(
                        "R2 S3 bucket policy status reports public access."
                    )
                checks[label] = (
                    "not_public"
                    if isinstance(status, dict) and status.get("IsPublic") is False
                    else "inconclusive"
                )
            elif label == "bucket_policy":
                if _policy_allows_public_access(response.get("Policy")):
                    raise R2PrivacyVerificationError(
                        "R2 S3 bucket policy allows a public principal."
                    )
                checks[label] = "not_public"
            else:
                grants = response.get("Grants", ())
                if not isinstance(grants, (list, tuple)):
                    raise R2PrivacyVerificationError(
                        "R2 S3 bucket ACL returned invalid grants."
                    )
                for grant in grants:
                    if not isinstance(grant, dict) or not isinstance(
                        grant.get("Grantee"), dict
                    ):
                        raise R2PrivacyVerificationError(
                            "R2 S3 bucket ACL contains an invalid grant."
                        )
                    grantee = grant["Grantee"]
                    uri = str(grantee.get("URI") or "").lower()
                    if any(value in uri for value in _PUBLIC_ACL_URIS):
                        raise R2PrivacyVerificationError(
                            "R2 S3 bucket ACL contains a public group grant."
                        )
                checks[label] = "not_public"
        return checks

    def _api_get(self, url: str, token: str, label: str) -> dict[str, Any]:
        if self.http_session is None:
            try:
                import requests
            except ModuleNotFoundError as exc:
                raise R2PrivacyVerificationUnavailable(
                    "requests is required for Cloudflare R2 privacy verification."
                ) from exc
            self.http_session = requests.Session()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "cf-r2-jurisdiction": self.config.jurisdiction,
        }
        try:
            response = self.http_session.get(
                url,
                headers=headers,
                timeout=30,
                allow_redirects=False,
            )
        except Exception as exc:
            raise R2PrivacyVerificationUnavailable(
                f"Cloudflare {label} privacy check failed ({type(exc).__name__})."
            ) from None
        if int(getattr(response, "status_code", 0)) != 200:
            raise R2PrivacyVerificationUnavailable(
                f"Cloudflare {label} privacy check returned HTTP "
                f"{int(getattr(response, 'status_code', 0))}."
            )
        try:
            payload = response.json()
        except Exception:
            raise R2PrivacyVerificationUnavailable(
                f"Cloudflare {label} privacy check returned invalid JSON."
            ) from None
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise R2PrivacyVerificationUnavailable(
                f"Cloudflare {label} privacy check did not report success."
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise R2PrivacyVerificationUnavailable(
                f"Cloudflare {label} privacy check has no result object."
            )
        return result

    def _verify_cloudflare_api(self, account_id: str, token: str) -> dict[str, Any]:
        root = (
            f"{_CLOUDFLARE_API_BASE}/accounts/{quote(account_id, safe='')}"
            f"/r2/buckets/{quote(self.config.bucket, safe='')}/domains"
        )
        managed = self._api_get(f"{root}/managed", token, "r2.dev")
        if (
            managed.get("enabled") is not False
            or not str(managed.get("domain") or "").strip().lower().endswith(".r2.dev")
        ):
            raise R2PrivacyVerificationError(
                "Cloudflare reports the R2 r2.dev domain as public or indeterminate."
            )
        custom_result = self._api_get(f"{root}/custom", token, "custom-domain")
        domains = custom_result.get("domains")
        if not isinstance(domains, list):
            raise R2PrivacyVerificationError(
                "Cloudflare custom-domain privacy check returned no domain list."
            )
        for domain in domains:
            if (
                not isinstance(domain, dict)
                or domain.get("enabled") is not False
                or not str(domain.get("domain") or "").strip()
            ):
                raise R2PrivacyVerificationError(
                    "Cloudflare reports an enabled or indeterminate R2 custom domain."
                )
        return {
            "status": "verified_private",
            "verification_method": "cloudflare_api",
            "managed_r2_dev_enabled": False,
            "custom_domain_count": len(domains),
            "enabled_custom_domain_count": 0,
            "checked_at": self.now().astimezone(timezone.utc).isoformat(),
        }

    def _verify_attestation(
        self,
        path: Path,
        expected_hash: str,
        *,
        endpoint_account: str,
        endpoint_host: str,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise R2PrivacyVerificationError("R2 privacy attestation file is missing.")
        content = path.read_bytes()
        if len(content) > 1_000_000:
            raise R2PrivacyVerificationError("R2 privacy attestation is unexpectedly large.")
        normalized_hash = expected_hash.strip().lower()
        if len(normalized_hash) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_hash
        ):
            raise R2PrivacyVerificationError(
                "R2 privacy attestation SHA-256 pin is invalid."
            )
        actual_hash = sha256_bytes(content)
        if not hmac.compare_digest(actual_hash, normalized_hash):
            raise R2PrivacyVerificationError("R2 privacy attestation hash mismatch.")
        try:
            value = json.loads(content)
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise R2PrivacyVerificationError(
                "R2 privacy attestation is invalid JSON."
            ) from exc
        if not isinstance(value, dict):
            raise R2PrivacyVerificationError(
                "R2 privacy attestation must be a JSON object."
            )
        method = str(value.get("verification_method") or "")
        if (
            value.get("schema_version") != 1
            or method not in {"cloudflare_api", "cloudflare_dashboard"}
            or str(value.get("account_id") or "") != endpoint_account
            or str(value.get("bucket") or "") != self.config.bucket
            or str(value.get("s3_endpoint_host") or "").lower() != endpoint_host
        ):
            raise R2PrivacyVerificationError(
                "R2 privacy attestation is not bound to this account, endpoint, and bucket."
            )
        managed = value.get("managed_domain")
        domains = value.get("custom_domains")
        if method == "cloudflare_api":
            if value.get("api_checks") != {
                "custom_domains": "passed",
                "managed_r2_dev": "passed",
            }:
                raise R2PrivacyVerificationError(
                    "R2 privacy attestation lacks exact Cloudflare API checks."
                )
            if (
                not isinstance(managed, dict)
                or managed.get("enabled") is not False
                or not str(managed.get("domain") or "")
                .strip()
                .lower()
                .endswith(".r2.dev")
            ):
                raise R2PrivacyVerificationError(
                    "R2 privacy attestation does not prove r2.dev is disabled."
                )
            if not isinstance(domains, list) or any(
                not isinstance(domain, dict)
                or domain.get("enabled") is not False
                or not str(domain.get("domain") or "").strip()
                for domain in domains
            ):
                raise R2PrivacyVerificationError(
                    "R2 privacy attestation contains an enabled or indeterminate custom domain."
                )
        else:
            evidence = value.get("dashboard_evidence")
            evidence_hash = (
                str(evidence.get("screenshot_sha256") or "").strip().lower()
                if isinstance(evidence, dict)
                else ""
            )
            if (
                value.get("dashboard_checks")
                != {"custom_domains": "passed", "managed_r2_dev": "passed"}
                or not isinstance(managed, dict)
                or managed
                != {"enabled": False, "state": "public_development_url_disabled"}
                or domains != []
                or not isinstance(evidence, dict)
                or evidence.get("kind") != "user_supplied_dashboard_screenshot"
                or len(evidence_hash) != 64
                or any(character not in "0123456789abcdef" for character in evidence_hash)
            ):
                raise R2PrivacyVerificationError(
                    "R2 dashboard attestation does not exactly prove private domain state."
                )
        checked_at = _parse_attestation_time(value.get("checked_at"), "checked_at")
        expires_at = _parse_attestation_time(value.get("expires_at"), "expires_at")
        now = self.now().astimezone(timezone.utc)
        max_age = self.config.privacy_attestation_max_age_seconds
        age = (now - checked_at).total_seconds()
        validity = (expires_at - checked_at).total_seconds()
        if age < -60 or age > max_age or expires_at < now or not 0 < validity <= max_age:
            raise R2PrivacyVerificationError(
                "R2 privacy attestation is stale, expired, or has excessive validity."
            )
        return {
            "status": "verified_private",
            "verification_method": "hash_pinned_attestation",
            "managed_r2_dev_enabled": False,
            "custom_domain_count": len(domains),
            "enabled_custom_domain_count": 0,
            "checked_at": checked_at.isoformat(),
            "attestation_sha256": actual_hash,
            "attestation_source": method,
        }


class R2ObjectStore:
    def __init__(self, config: R2Config):
        if not config.enabled:
            raise ValueError("R2 is disabled in configuration.")
        load_env()
        try:
            import boto3
        except ModuleNotFoundError as exc:
            raise RuntimeError("boto3 is required for Cloudflare R2 support.") from exc
        access_key = os.getenv(config.access_key_env)
        secret_key = os.getenv(config.secret_key_env)
        endpoint_url = os.getenv(config.endpoint_env)
        if not endpoint_url or not access_key or not secret_key:
            raise RuntimeError(
                "R2 environment values are missing: "
                f"{config.endpoint_env}, {config.access_key_env}, {config.secret_key_env}"
            )
        self.bucket = config.bucket
        self.prefix = config.prefix.strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=config.region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._privacy_verifier = R2PrivacyVerifier(
            config,
            self.client,
            endpoint_url,
        )
        self._privacy_lock = threading.Lock()
        self._privacy_verification: dict[str, Any] | None = None

    def verify_private_access(self, *, force: bool = False) -> dict[str, Any]:
        """Prove actual private visibility and cache it for this process."""

        with self._privacy_lock:
            if force:
                self._privacy_verification = None
            if self._privacy_verification is None or force:
                result = self._privacy_verifier.verify()
                if (
                    not isinstance(result, dict)
                    or result.get("status") != "verified_private"
                    or result.get("verification_method")
                    not in {"cloudflare_api", "hash_pinned_attestation"}
                    or result.get("managed_r2_dev_enabled") is not False
                    or type(result.get("enabled_custom_domain_count")) is not int
                    or result.get("enabled_custom_domain_count") != 0
                    or not isinstance(result.get("s3_checks"), dict)
                    or result["s3_checks"].get("head_bucket") != "passed"
                ):
                    raise R2PrivacyVerificationError(
                        "R2 private-state verifier returned an invalid result."
                    )
                self._privacy_verification = result
            return dict(self._privacy_verification)

    def _key(self, key: str) -> str:
        return "/".join(part for part in (self.prefix, key.lstrip("/")) if part)

    def get(self, key: str) -> ObjectValue:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except Exception as exc:
            if _client_error_code(exc) in {"NoSuchKey", "404", "NotFound"}:
                raise ObjectNotFound(key) from exc
            raise
        return ObjectValue(
            data=response["Body"].read(),
            etag=str(response.get("ETag", "")).strip('"'),
        )

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
    ) -> str:
        kwargs: dict[str, object] = {
            "Bucket": self.bucket,
            "Key": self._key(key),
            "Body": data,
        }
        if if_match is not None:
            kwargs["IfMatch"] = if_match
        if if_none_match:
            kwargs["IfNoneMatch"] = "*"
        # This is intentionally immediately before the first mutating S3 call.
        # All R2 write paths therefore fail closed even if they bypass the
        # operator publication script.
        self.verify_private_access()
        try:
            response = self.client.put_object(**kwargs)
        except Exception as exc:
            if _client_error_code(exc) in {"PreconditionFailed", "412", "ConditionalRequestConflict"}:
                raise ConditionalWriteFailed(f"Conditional R2 write failed: {key}") from exc
            raise
        return str(response.get("ETag", "")).strip('"')

    def list(self, prefix: str) -> tuple[str, ...]:
        remote_prefix = self._key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        values: list[str] = []
        root_prefix = f"{self.prefix}/" if self.prefix else ""
        for page in paginator.paginate(Bucket=self.bucket, Prefix=remote_prefix):
            for item in page.get("Contents", ()):
                key = str(item["Key"])
                values.append(key.removeprefix(root_prefix))
        return tuple(values)


@dataclass(frozen=True)
class PublishResult:
    pointer: CurrentPointer
    pointer_etag: str
    conflict: bool = False
    conflict_prefix: str = ""


class DatasetPublisher:
    def __init__(self, store: ObjectStore):
        self.store = store

    @staticmethod
    def version_prefix(dataset: str, version: str) -> str:
        return f"datasets/{dataset}/versions/{version}"

    @staticmethod
    def current_key(dataset: str) -> str:
        return f"datasets/{dataset}/current.json"

    def current(self, dataset: str) -> tuple[CurrentPointer | None, str | None]:
        try:
            value = self.store.get(self.current_key(dataset))
        except ObjectNotFound:
            return None, None
        return CurrentPointer.from_bytes(value.data), value.etag

    def publish(
        self,
        local_version_root: str | Path,
        manifest: DatasetManifest,
        *,
        expected_pointer_etag: str | None,
    ) -> PublishResult:
        root = Path(local_version_root)
        prefix = self.version_prefix(manifest.dataset, manifest.version)
        for item in manifest.files:
            self._put_immutable(f"{prefix}/{item.path}", (root / item.path).read_bytes())
        manifest_path = f"{prefix}/manifest.json"
        manifest_bytes = manifest.to_bytes()
        self._put_immutable(manifest_path, manifest_bytes)
        pointer = CurrentPointer.create(manifest, manifest_path)
        try:
            pointer_etag = self.store.put(
                self.current_key(manifest.dataset),
                pointer.to_bytes(),
                if_match=expected_pointer_etag,
                if_none_match=expected_pointer_etag is None,
            )
        except ConditionalWriteFailed:
            conflict_prefix = f"conflicts/{manifest.dataset}/{manifest.version}"
            self.store.put(f"{conflict_prefix}/manifest.json", manifest_bytes, if_none_match=True)
            return PublishResult(pointer, "", conflict=True, conflict_prefix=conflict_prefix)
        return PublishResult(pointer, pointer_etag)

    def upload_version(
        self,
        local_version_root: str | Path,
        manifest: DatasetManifest,
    ) -> None:
        root = Path(local_version_root)
        prefix = self.version_prefix(manifest.dataset, manifest.version)
        for item in manifest.files:
            self._put_immutable(f"{prefix}/{item.path}", (root / item.path).read_bytes())
        self._put_immutable(f"{prefix}/manifest.json", manifest.to_bytes())

    def advance_current(
        self,
        manifest: DatasetManifest,
        *,
        expected_pointer_etag: str | None,
    ) -> PublishResult:
        manifest_path = f"{self.version_prefix(manifest.dataset, manifest.version)}/manifest.json"
        pointer = CurrentPointer.create(manifest, manifest_path)
        try:
            etag = self.store.put(
                self.current_key(manifest.dataset),
                pointer.to_bytes(),
                if_match=expected_pointer_etag,
                if_none_match=expected_pointer_etag is None,
            )
        except ConditionalWriteFailed:
            conflict_prefix = f"conflicts/{manifest.dataset}/{manifest.version}"
            self._put_immutable(f"{conflict_prefix}/manifest.json", manifest.to_bytes())
            return PublishResult(pointer, "", conflict=True, conflict_prefix=conflict_prefix)
        return PublishResult(pointer, etag)

    def _put_immutable(self, key: str, data: bytes) -> None:
        try:
            self.store.put(key, data, if_none_match=True)
        except ConditionalWriteFailed:
            existing = self.store.get(key)
            if existing.data != data:
                raise ConditionalWriteFailed(f"Immutable object differs: {key}")


class DatasetCache:
    def __init__(self, root: str | Path, store: ObjectStore):
        self.root = Path(root)
        self.store = store

    def sync(self, dataset: str) -> DatasetManifest:
        pointer_value = self.store.get(DatasetPublisher.current_key(dataset))
        pointer = CurrentPointer.from_bytes(pointer_value.data)
        manifest_value = self.store.get(pointer.manifest_path)
        if sha256_bytes(manifest_value.data) != pointer.manifest_sha256:
            raise ValueError(f"Remote manifest hash mismatch for {dataset}")
        manifest = DatasetManifest.from_bytes(manifest_value.data)
        self._sync_manifest_chain(dataset, manifest, manifest_value.data, set())
        write_atomic(self.root / "datasets" / dataset / "current.json", pointer_value.data)
        if dataset == "source_archive":
            self._sync_archive_payloads(manifest)
        return manifest

    def sync_release(self, datasets: tuple[str, ...] | None = None) -> DataRelease:
        release_value = self.store.get("releases/current.json")
        release = DataRelease.from_bytes(release_value.data)
        selected = set(datasets or release.dataset_versions)
        for dataset, version in release.dataset_versions.items():
            if dataset not in selected:
                continue
            manifest_path = f"{DatasetPublisher.version_prefix(dataset, version)}/manifest.json"
            manifest_value = self.store.get(manifest_path)
            manifest = DatasetManifest.from_bytes(manifest_value.data)
            self._sync_manifest_chain(dataset, manifest, manifest_value.data, set())
            pointer = CurrentPointer.create(manifest, manifest_path)
            write_atomic(self.root / "datasets" / dataset / "current.json", pointer.to_bytes())
            if dataset == "source_archive":
                self._sync_archive_payloads(manifest)
        immutable_value = self.store.get(f"releases/{release.version}.json")
        if immutable_value.data != release_value.data:
            raise ValueError(f"Release current/immutable mismatch: {release.version}")
        if datasets is None or set(release.dataset_versions).issubset(selected):
            write_atomic(self.root / f"releases/{release.version}.json", immutable_value.data)
            write_atomic(self.root / "releases/current.json", release_value.data)
        return release

    def _sync_manifest_chain(
        self,
        dataset: str,
        manifest: DatasetManifest,
        manifest_bytes: bytes,
        seen: set[str],
    ) -> None:
        if manifest.version in seen:
            raise ValueError(f"Remote manifest cycle detected: {dataset}/{manifest.version}")
        seen.add(manifest.version)
        if bool(manifest.metadata.get("inherits_parent")):
            if not manifest.parent_version:
                raise ValueError(f"Remote inherited version has no parent: {dataset}/{manifest.version}")
            parent_path = (
                f"{DatasetPublisher.version_prefix(dataset, manifest.parent_version)}/manifest.json"
            )
            parent_value = self.store.get(parent_path)
            parent = DatasetManifest.from_bytes(parent_value.data)
            self._sync_manifest_chain(dataset, parent, parent_value.data, seen)
        version_root = self.root / "datasets" / dataset / "versions" / manifest.version
        for item in manifest.files:
            local = version_root / item.path
            if not local.is_file() or sha256_bytes(local.read_bytes()) != item.sha256:
                remote_key = f"{DatasetPublisher.version_prefix(dataset, manifest.version)}/{item.path}"
                value = self.store.get(remote_key)
                if sha256_bytes(value.data) != item.sha256:
                    raise ValueError(f"Remote file hash mismatch: {remote_key}")
                write_atomic(local, value.data)
        write_atomic(version_root / "manifest.json", manifest_bytes)

    def _sync_archive_payloads(self, manifest: DatasetManifest) -> None:
        try:
            import pandas as pd
        except ModuleNotFoundError:
            return
        paths = []
        current = manifest
        while True:
            root = self.root / "datasets" / current.dataset / "versions" / current.version
            paths.extend(root / item.path for item in current.files)
            if not bool(current.metadata.get("inherits_parent")):
                break
            parent_path = root.parent / current.parent_version / "manifest.json"
            current = DatasetManifest.from_bytes(parent_path.read_bytes())
        if not paths:
            return
        frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
        if "object_path" not in frame:
            return
        for object_path in frame["object_path"].dropna().astype(str).drop_duplicates():
            local = _safe_cache_object_path(self.root, object_path)
            if local.is_file():
                continue
            value = self.store.get(object_path)
            write_atomic(local, value.data)


def _safe_cache_object_path(root: str | Path, object_path: str) -> Path:
    """Resolve an archive object below a cache root before any remote read."""

    resolved_root = Path(root).resolve()
    resolved = (resolved_root / object_path).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"Archive object path escapes cache root: {object_path}")
    return resolved


@dataclass(frozen=True)
class RepositoryPublishResult:
    dataset: str
    version: str
    published: bool
    conflict: bool = False
    detail: str = ""


def publish_repository(
    repository,
    store: ObjectStore,
    datasets: tuple[str, ...],
    *,
    supersede_versions: Mapping[str, str] | None = None,
) -> tuple[RepositoryPublishResult, ...]:
    publisher = DatasetPublisher(store)
    output: list[RepositoryPublishResult] = []
    original_release, _ = repository.current_release()
    for dataset in datasets:
        publish_detail = ""
        local = repository.current_manifest(dataset)
        if local is None:
            output.append(RepositoryPublishResult(dataset, "", False, detail="local dataset missing"))
            continue
        remote, remote_etag = publisher.current(dataset)
        chain = repository.manifest_chain(dataset, local.version)
        chain_versions = {item.version for item in chain}
        if remote is not None and remote.version == local.version:
            # A previous interrupted publication may have advanced the current
            # pointer after writing the leaf manifest while an older inherited
            # manifest or archive payload is still absent.  Reconcile the full
            # immutable lineage before treating an equal pointer as complete.
            for manifest in chain:
                root = repository.root / repository.version_prefix(
                    dataset, manifest.version
                )
                publisher.upload_version(root, manifest)
            if dataset == "source_archive":
                archive_frame = repository.read_frame(dataset)
                for object_path in archive_frame.get("object_path", ()):
                    path = repository.root / str(object_path)
                    if path.is_file():
                        publisher._put_immutable(str(object_path), path.read_bytes())
            output.append(
                RepositoryPublishResult(
                    dataset,
                    local.version,
                    False,
                    detail="already current; immutable lineage reconciled",
                )
            )
            continue
        allowed_supersede = str((supersede_versions or {}).get(dataset, ""))
        if (
            remote is not None
            and remote.version not in chain_versions
            and remote.version == allowed_supersede
        ):
            # The operator explicitly bound this overwrite to the immutable
            # dataset version referenced by the previously validated remote
            # release. Preserve every old object, upload the complete new
            # lineage, and use the pointer ETag so a concurrent writer still
            # wins safely.
            for manifest in chain:
                root = repository.root / repository.version_prefix(
                    dataset, manifest.version
                )
                publisher.upload_version(root, manifest)
            if dataset == "source_archive":
                archive_frame = repository.read_frame(dataset)
                for object_path in archive_frame.get("object_path", ()):
                    path = repository.root / str(object_path)
                    if path.is_file():
                        publisher._put_immutable(str(object_path), path.read_bytes())
            advanced = publisher.advance_current(
                local,
                expected_pointer_etag=remote_etag,
            )
            output.append(
                RepositoryPublishResult(
                    dataset,
                    local.version,
                    not advanced.conflict,
                    conflict=advanced.conflict,
                    detail=(
                        advanced.conflict_prefix
                        or f"superseded remote release version {allowed_supersede}"
                    ),
                )
            )
            continue
        if remote is not None and remote.version not in chain_versions:
            remote_manifest = DatasetManifest.from_bytes(
                store.get(remote.manifest_path).data
            )
            merged, merge_detail = _merge_divergent_dataset(
                repository,
                store,
                dataset,
                local,
                remote_manifest,
            )
            if merged is None:
                conflict_key = f"conflicts/{dataset}/{local.version}/manifest.json"
                publisher._put_immutable(conflict_key, local.to_bytes())
                try:
                    repository.objects.put(
                        conflict_key,
                        local.to_bytes(),
                        if_none_match=True,
                    )
                except ConditionalWriteFailed:
                    pass
                output.append(
                    RepositoryPublishResult(
                        dataset,
                        local.version,
                        False,
                        conflict=True,
                        detail=merge_detail,
                    )
                )
                continue
            local = merged
            publish_detail = merge_detail
            chain = repository.manifest_chain(dataset, local.version)
            chain_versions = {item.version for item in chain}
            if local.version == remote.version:
                output.append(
                    RepositoryPublishResult(
                        dataset,
                        local.version,
                        False,
                        detail=merge_detail or "remote already contains local changes",
                    )
                )
                continue
        upload = remote is None
        for manifest in chain:
            if not upload and manifest.version == remote.version:
                upload = True
                continue
            if upload:
                root = repository.root / repository.version_prefix(dataset, manifest.version)
                publisher.upload_version(root, manifest)
        if dataset == "source_archive":
            archive_frame = repository.read_frame(dataset)
            for object_path in archive_frame.get("object_path", ()):
                path = repository.root / str(object_path)
                if path.is_file():
                    publisher._put_immutable(str(object_path), path.read_bytes())
        advanced = publisher.advance_current(local, expected_pointer_etag=remote_etag)
        output.append(
            RepositoryPublishResult(
                dataset,
                local.version,
                not advanced.conflict,
                conflict=advanced.conflict,
                detail=advanced.conflict_prefix or publish_detail,
            )
        )
    local_release = original_release
    if local_release is not None and not any(item.conflict for item in output):
        merged_versions = dict(local_release.dataset_versions)
        for item in output:
            if item.dataset != "__release__" and item.version and not item.conflict:
                merged_versions[item.dataset] = item.version
        if merged_versions != local_release.dataset_versions:
            completed_session = max(
                repository.manifest_for_version(dataset, version).completed_session
                for dataset, version in merged_versions.items()
            )
            local_release = repository.commit_release(
                completed_session,
                merged_versions,
                quality=local_release.quality,
                warnings=local_release.warnings,
            )
    if local_release is not None:
        dataset_conflict = any(item.conflict for item in output)
        remote_mismatches: list[str] = []
        if not dataset_conflict:
            for dataset, version in local_release.dataset_versions.items():
                remote_manifest, _ = publisher.current(dataset)
                if remote_manifest is None:
                    remote_mismatches.append(f"{dataset}=missing (expected {version})")
                elif remote_manifest.version != version:
                    remote_mismatches.append(
                        f"{dataset}={remote_manifest.version} (expected {version})"
                    )
        if dataset_conflict or remote_mismatches:
            detail = "dataset conflict prevented release publication"
            if remote_mismatches:
                detail = "release datasets are not current remotely: " + ", ".join(remote_mismatches)
            output.append(
                RepositoryPublishResult(
                    "__release__",
                    local_release.version,
                    False,
                    conflict=True,
                    detail=detail,
                )
            )
        else:
            try:
                remote_release_value = store.get("releases/current.json")
                remote_release = DataRelease.from_bytes(remote_release_value.data)
                remote_release_etag = remote_release_value.etag
            except ObjectNotFound:
                remote_release = None
                remote_release_etag = None
            if remote_release is not None and remote_release.version == local_release.version:
                output.append(
                    RepositoryPublishResult(
                        "__release__", local_release.version, False, detail="already current"
                    )
                )
            else:
                immutable_key = f"releases/{local_release.version}.json"
                release_bytes = local_release.to_bytes()
                publisher._put_immutable(immutable_key, release_bytes)
                try:
                    store.put(
                        "releases/current.json",
                        release_bytes,
                        if_match=remote_release_etag,
                        if_none_match=remote_release_etag is None,
                    )
                except ConditionalWriteFailed:
                    publisher._put_immutable(
                        f"conflicts/releases/{local_release.version}.json",
                        release_bytes,
                    )
                    output.append(
                        RepositoryPublishResult(
                            "__release__",
                            local_release.version,
                            False,
                            conflict=True,
                            detail="release CAS failed",
                        )
                    )
                else:
                    output.append(
                        RepositoryPublishResult("__release__", local_release.version, True)
                    )
    return tuple(output)


def _merge_divergent_dataset(
    repository,
    store: ObjectStore,
    dataset: str,
    local: DatasetManifest,
    remote: DatasetManifest,
) -> tuple[DatasetManifest | None, str]:
    """Rebase append-only disjoint changes; conflicting values remain quarantined."""
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        return None, f"cannot merge divergent versions without pandas: {exc}"

    local_lineage = _local_lineage(repository, dataset, local)
    remote_lineage = _remote_lineage_with_local_repair(
        repository,
        store,
        dataset,
        remote,
    )
    remote_versions = {manifest.version for manifest in remote_lineage}
    common = next(
        (manifest for manifest in local_lineage if manifest.version in remote_versions),
        None,
    )
    if common is None:
        return None, f"remote current {remote.version} has no common ancestor"

    local_frame = repository.read_frame(dataset, local.version)
    base_frame = repository.read_frame(dataset, common.version)
    with tempfile.TemporaryDirectory(prefix="stq-remote-merge-", dir=repository.root) as directory:
        remote_repository = repository.__class__(directory)
        DatasetCache(directory, store).sync(dataset)
        remote_frame = remote_repository.read_frame(dataset, remote.version)

    # The repository exposes the schema through its module-level dataset_spec;
    # infer the logical key from the already validated frames to avoid a storage
    # -> repository import cycle.
    primary_key = dataset_spec(dataset).primary_key
    local_changes, local_deletes = _frame_changes(
        dataset, base_frame, local_frame, primary_key
    )
    remote_changes, remote_deletes = _frame_changes(
        dataset, base_frame, remote_frame, primary_key
    )
    if local_deletes or remote_deletes:
        return None, "divergent snapshots contain deletions and require manual conflict review"

    overlap = set(local_changes) & set(remote_changes)
    conflicts = [
        key
        for key in overlap
        if _business_record(dataset, local_changes[key])
        != _business_record(dataset, remote_changes[key])
    ]
    if conflicts:
        return None, f"same-key value conflict on {len(conflicts)} row(s)"

    delta_records = [
        record
        for key, record in local_changes.items()
        if key not in remote_changes
    ]
    DatasetCache(repository.root, store).sync(dataset)
    if not delta_records:
        return repository.current_manifest(dataset), "remote already contains equivalent changes"
    delta = pd.DataFrame(delta_records)
    result = repository.append_frame(
        dataset,
        delta,
        completed_session=max(local.completed_session, remote.completed_session),
        metadata={
            "operation": "merge_disjoint_publishers",
            "merged_local_version": local.version,
            "merged_remote_version": remote.version,
        },
    )
    if result.conflict:
        return None, f"merge current-pointer CAS failed: {result.conflict_path}"
    return result.manifest, "disjoint changes automatically rebased"


def _local_lineage(repository, dataset: str, latest: DatasetManifest) -> list[DatasetManifest]:
    output = [latest]
    seen = {latest.version}
    current = latest
    while current.parent_version:
        if current.parent_version in seen:
            raise ValueError(f"Dataset manifest cycle detected: {dataset}/{current.parent_version}")
        current = repository.manifest_for_version(dataset, current.parent_version)
        output.append(current)
        seen.add(current.version)
    return output


def _remote_lineage(
    store: ObjectStore,
    dataset: str,
    latest: DatasetManifest,
) -> list[DatasetManifest]:
    output = [latest]
    seen = {latest.version}
    current = latest
    while current.parent_version:
        if current.parent_version in seen:
            raise ValueError(f"Remote manifest cycle detected: {dataset}/{current.parent_version}")
        key = f"{DatasetPublisher.version_prefix(dataset, current.parent_version)}/manifest.json"
        current = DatasetManifest.from_bytes(store.get(key).data)
        output.append(current)
        seen.add(current.version)
    return output


def _remote_lineage_with_local_repair(
    repository,
    store: ObjectStore,
    dataset: str,
    latest: DatasetManifest,
) -> list[DatasetManifest]:
    """Repair interrupted immutable-parent uploads before a divergent merge.

    A publisher can be interrupted after advancing a child pointer but before
    every inherited parent manifest is present remotely.  If the missing
    parent is part of this publisher's validated local repository, restore
    that one immutable version and retry the lineage read.  Missing versions
    that are not available locally still fail closed.
    """

    publisher = DatasetPublisher(store)
    repaired: set[str] = set()
    while True:
        try:
            return _remote_lineage(store, dataset, latest)
        except ObjectNotFound as exc:
            key = str(exc)
            prefix = f"datasets/{dataset}/versions/"
            suffix = "/manifest.json"
            if not key.startswith(prefix) or not key.endswith(suffix):
                raise
            version = key[len(prefix) : -len(suffix)]
            if not version or version in repaired:
                raise
            try:
                manifest = repository.manifest_for_version(dataset, version)
            except (FileNotFoundError, ObjectNotFound):
                raise exc
            root = repository.root / repository.version_prefix(dataset, version)
            publisher.upload_version(root, manifest)
            repaired.add(version)


def _frame_changes(dataset: str, base, candidate, primary_key: tuple[str, ...]):
    base_records = _record_map(base, primary_key)
    candidate_records = _record_map(candidate, primary_key)
    changed = {
        key: record
        for key, record in candidate_records.items()
        if key not in base_records
        or _business_record(dataset, record)
        != _business_record(dataset, base_records[key])
    }
    deleted = set(base_records) - set(candidate_records)
    return changed, deleted


def _record_map(frame, primary_key: tuple[str, ...]) -> dict[tuple[str, ...], dict]:
    output = {}
    for record in frame.to_dict("records"):
        key = tuple(str(record[column]) for column in primary_key)
        output[key] = record
    return output


def _business_record(dataset: str, record: dict) -> dict:
    ignored = {"source", "source_url", "source_kind", "retrieved_at", "source_hash"}
    if dataset == "adjustment_factors":
        ignored.update({"source_version", "calculated_at"})
    return {
        key: _canonical_value(value)
        for key, value in record.items()
        if key not in ignored
    }


def _canonical_value(value):
    try:
        import pandas as pd

        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except (ModuleNotFoundError, TypeError, ValueError):
        pass
    if isinstance(value, (dict, list, tuple)):
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value




def _client_error_code(exc: Exception) -> str:
    response = getattr(exc, "response", {})
    if not isinstance(response, dict):
        return ""
    error = response.get("Error", {})
    return str(error.get("Code", "")) if isinstance(error, dict) else ""
