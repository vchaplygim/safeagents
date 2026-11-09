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

from collections.abc import Iterable

from .constants import (
    AUTHORITY_PATTERN,
    DEFAULT_ALLOWED_HOSTS,
    FILE_WRITE_TOOL,
    HOST_PATTERN,
    IPV4_PATTERN,
    UPLOAD_PATTERN,
)
from .utils import content_text


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


def sourced_hosts(request: dict) -> set[str]:
    """Hosts the human put in play: those named in the user, system and developer turns.

    Assistant turns and tool results are excluded on purpose. A host the model wrote
    earlier is no more sourced than one it writes now, and a host that only appears in
    a tool's output can be laundered into one by a call that echoes it back.
    """
    hosts: set[str] = set()
    hosts.update(extract_hosts(content_text(request.get("system"))))
    for message in request.get("messages") or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") in ("user", "system", "developer"):
            # content_text keeps only text/input_text blocks, so Anthropic tool_result
            # blocks carried on a user message are dropped here.
            hosts.update(extract_hosts(content_text(message.get("content"))))
    return hosts


def host_allowed(host: str, allowed: Iterable[str]) -> bool:
    """`host` equals one of `allowed`, or is a subdomain of one."""
    return any(host == entry or host.endswith("." + entry) for entry in allowed)


class Policy:
    """What a request allows its response to call."""

    def __init__(
        self,
        request: dict,
        allowed_hosts: Iterable[str] = DEFAULT_ALLOWED_HOSTS,
        upload_hosts: Iterable[str] = (),
    ):
        self.declared = declared_tools(request)
        self.sourced = sourced_hosts(request)
        self.allowed_hosts = [host.lower() for host in allowed_hosts]
        # Hosts a call may send data to. The GET allowlist (package registries and the
        # like) does not grant uploads, and merely naming a host in the chat does not
        # authorise exfiltrating to it, so this list is separate and empty by default.
        self.upload_hosts = [host.lower() for host in upload_hosts]

    def check(self, name: str, args: str) -> str | None:
        """Why a call to `name` with raw `args` should be blocked, if it should."""
        if name not in self.declared:
            return "undeclared tool"
        # Writing a host into a file contacts nothing; the host rule is about reaching one.
        if FILE_WRITE_TOOL.match(name):
            return None
        uploading = bool(UPLOAD_PATTERN.search(args))
        for host in extract_hosts(args):
            if uploading:
                if host_allowed(host, self.upload_hosts):
                    continue
                return f"host `{host}` receives uploaded data but is not an allowed upload target"
            if host_allowed(host, self.sourced) or host_allowed(host, self.allowed_hosts):
                continue
            return f"host `{host}` appears nowhere in the conversation"
        return None
