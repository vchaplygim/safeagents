# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checks that need only the request: the tools it declared and the hosts it mentions."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Iterable

from .constants import (
    AUTHORITY_PATTERN,
    COPY_TOOL_PATTERN,
    CURL_UPLOAD_PATTERN,
    DEFAULT_ALLOWED_HOSTS,
    HOST_PATTERN,
    IPV4_PATTERN,
    NETWORK_PATTERN,
    NETWORK_TOOL_PATTERN,
    REMOTE_SPEC_PATTERN,
    RESERVED_HOSTS,
    SEGMENT_SPLIT,
    WGET_UPLOAD_PATTERN,
)


def declared_tools(request: dict) -> set[str]:
    """Tool names in the request's `tools`: OpenAI `function.name`, Anthropic `name`."""
    names = set()
    for tool in request.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else tool.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def extract_hosts(text: str) -> list[str]:
    """Lowercased hosts in `text`: the host of `scheme://` URLs and bare IPv4 literals."""
    text = text.lower()
    hosts = set()
    for authority in AUTHORITY_PATTERN.findall(text):
        host_and_port = authority.rsplit("@", 1)[-1]
        host = HOST_PATTERN.match(host_and_port).group()
        if host:
            hosts.add(host)
    for ip_address in IPV4_PATTERN.findall(text):
        if all(int(octet) <= 255 for octet in ip_address.split(".")):
            hosts.add(ip_address)
    return sorted(hosts)


def call_text(args: str) -> str:
    """The text a call's raw JSON arguments carry: every string value, decoded (a command's newlines are real again).

    Matching the raw JSON instead reads an escaped newline before `cat` as `ncat`, a network client."""
    try:
        value = json.loads(args)
    except ValueError:
        return args
    strings = []

    def walk(item):
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            for nested in item.values():
                walk(nested)
        elif isinstance(item, list):
            for nested in item:
                walk(nested)

    walk(value)
    return "\n".join(strings)


def host_allowed(host: str, allowed: Iterable[str]) -> bool:
    """`host` equals one of `allowed`, or is a subdomain of one."""
    return any(host == entry or host.endswith("." + entry) for entry in allowed)


def is_loopback(host: str) -> bool:
    """`localhost`, a `.localhost` name, or a loopback/unspecified IP literal."""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def segment_targets(segment: str) -> tuple[list[str], list[str]]:
    """The scp/rsync remotes one shell segment reaches, and the hosts it uploads to.

    URL hosts a segment merely reaches are left to the host rule; this adds what it cannot
    see (`user@host:path` remotes) and what it does not tell apart (a request with a body)."""
    if CURL_UPLOAD_PATTERN.search(segment) or WGET_UPLOAD_PATTERN.search(segment):
        return [], extract_hosts(segment)
    copy = COPY_TOOL_PATTERN.search(segment)
    if copy is None:
        return [], []
    operands = [token.strip("\"'") for token in segment[copy.end() :].split() if not token.startswith("-")]
    reached, uploaded = [], []
    for index, operand in enumerate(operands):
        remote = REMOTE_SPEC_PATTERN.match(operand)
        if remote is not None:
            # The last operand is the destination: a remote one makes this an upload.
            (uploaded if index == len(operands) - 1 else reached).append(remote.group("host").lower())
    return reached, uploaded


class Policy:
    """What a request allows its response to call."""

    def __init__(
        self,
        request: dict,
        allowed_hosts: Iterable[str] = DEFAULT_ALLOWED_HOSTS,
        upload_hosts: Iterable[str] = (),
    ):
        self.declared = declared_tools(request)
        # The request without the model's own turns: a host the model wrote earlier
        # is no more sourced than one it writes now.
        context = dict(request)
        if isinstance(context.get("messages"), list):
            context["messages"] = [
                message
                for message in context["messages"]
                if not (isinstance(message, dict) and message.get("role") == "assistant")
            ]
        self.context = json.dumps(context, ensure_ascii=False).lower()
        self.allowed_hosts = [host.lower() for host in (*allowed_hosts, *RESERVED_HOSTS)]
        # Hosts a call may send data to. Being allowed to reach a host, or seeing it named in the conversation, does
        # not allow sending data there, so this list is separate and empty by default.
        self.upload_hosts = [host.lower() for host in upload_hosts]

    def check(self, name: str, args: str) -> str | None:
        """Why a call to `name` with raw `args` should be blocked, if it should."""
        if name not in self.declared:
            return "undeclared tool"
        # A host only matters in a call that can reach it: one in a file the call writes is data
        text = call_text(args)
        network_tool = NETWORK_TOOL_PATTERN.search(name)
        if not (network_tool or NETWORK_PATTERN.search(text)):
            return None
        # Sending data is judged per shell segment, so a POST to a local server does not turn a GET elsewhere in the
        # same command into an upload.
        remotes: set[str] = set()
        uploaded: set[str] = set()
        for segment in SEGMENT_SPLIT.split(text):
            segment_reached, segment_uploaded = segment_targets(segment)
            remotes.update(segment_reached)
            uploaded.update(segment_uploaded)
        for host in sorted(uploaded):
            if not (is_loopback(host) or host_allowed(host, self.upload_hosts)):
                return f"host `{host}` receives uploaded data but is not an allowed upload target"
        # A URL's host counts anywhere in such a call (code often names the URL a line before it opens it). A bare IP
        # counts only on a line that reaches out: elsewhere it is data, like the log lines of a test fixture.
        reaching = (
            text if network_tool else "\n".join(line for line in text.splitlines() if NETWORK_PATTERN.search(line))
        )
        url_hosts = [host for host in extract_hosts(text) if not IPV4_PATTERN.fullmatch(host)]
        # A host cleared for uploads can be reached too: sending data there means connecting to it.
        for host in sorted({*url_hosts, *extract_hosts(reaching), *remotes} - uploaded):
            if host in self.context:
                continue
            if host_allowed(host, self.allowed_hosts):
                continue
            return f"host `{host}` appears nowhere in the conversation"
        return None
