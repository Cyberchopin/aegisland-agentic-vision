from __future__ import annotations

import base64
import json
import os
import time
from decimal import Decimal
from typing import Any

import boto3

TABLE_NAME = os.environ["TRACE_TABLE"]
BUCKET_NAME = os.environ["EVIDENCE_BUCKET"]
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")
table = dynamodb.Table(TABLE_NAME)


def _response(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload),
    }


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    if event.get("requestContext", {}).get("http", {}).get("method") != "POST":
        return _response(405, {"error": "method_not_allowed"})
    try:
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        if len(body.encode("utf-8")) > 256_000:
            return _response(413, {"error": "trace_too_large"})
        payload = json.loads(body, parse_float=Decimal)
        trace_id = str(payload["trace_id"])
        sequence = int(payload["sequence"])
        if not trace_id.isalnum() or len(trace_id) > 64 or sequence < 0:
            raise ValueError("invalid trace identity")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _response(400, {"error": "invalid_trace", "detail": str(exc)})

    received_at = int(time.time() * 1000)
    item = {
        "TraceId": trace_id,
        "Sequence": sequence,
        "ReceivedAt": received_at,
        "Action": str(payload.get("decision", {}).get("action", "unknown")),
        "RiskScore": payload.get("decision", {}).get("risk_score", Decimal(0)),
        "EvidenceId": str(payload.get("evidence", {}).get("evidence_id", "unknown")),
        "ExpiresAt": int(time.time()) + 60 * 60 * 24 * 30,
    }
    # ConditionExpression makes client retries idempotent instead of duplicating evidence.
    try:
        table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(TraceId) AND attribute_not_exists(Sequence)",
        )
    except table.meta.client.exceptions.ConditionalCheckFailedException:
        return _response(200, {"status": "duplicate_ignored", "trace_id": trace_id, "sequence": sequence})

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"traces/{trace_id}/{sequence:08d}.json",
        Body=body.encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
    return _response(202, {"status": "accepted", "trace_id": trace_id, "sequence": sequence})

