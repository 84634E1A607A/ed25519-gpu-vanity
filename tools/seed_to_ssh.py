#!/usr/bin/env python3
"""Convert a raw ed25519 seed (as printed by cuda_ed25519_vanity MATCH lines)
into an OpenSSH keypair.

The miner prints, for every match:
    GPU <n> MATCH ,<64-char seed hex>
    [<64-char public key hex>]

Usage:
    python3 tools/seed_to_ssh.py <seed-hex> [outdir]

Writes <outdir>/id_ed25519_vanity (0600) and id_ed25519_vanity.pub (0644)
and verifies them with ssh-keygen. Only python stdlib is required;
ssh-keygen is used for the final independent verification.
"""
import base64
import os
import struct
import subprocess
import sys

def ssh_string(b: bytes) -> bytes:
    return struct.pack(">I", len(b)) + b

def seed_to_keypair(seed_hex: str, outdir: str, comment: str = "") -> None:
    seed = bytes.fromhex(seed_hex)
    if len(seed) != 32:
        raise SystemExit("seed must be exactly 32 bytes (64 hex chars)")

    # Independent seed -> pubkey derivation (python cryptography if present).
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        pub = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        print("pubkey (cryptography):", pub.hex())
    except ImportError:
        pub = None
        print("python cryptography not available; skipping cross-check")

    # OpenSSH wire blob: string("ssh-ed25519") || string(32-byte pubkey) = 51 bytes.
    blob = b"\x00\x00\x00\x0bssh-ed25519" + b"\x00\x00\x00\x20" + pub
    b64 = base64.b64encode(blob).decode()
    assert len(b64) == 68

    # openssh-key-v1 private key container, cipher/kdf "none".
    checkint = os.urandom(4)
    priv_body = (
        checkint + checkint
        + ssh_string(b"ssh-ed25519")
        + ssh_string(pub)
        + ssh_string(seed + pub)      # ed25519 "private" = seed(32) || pub(32)
        + ssh_string(comment.encode())
    )
    priv_body += bytes(range(1, (8 - len(priv_body) % 8) % 8 + 1))

    inner = (
        b"openssh-key-v1\x00"
        + ssh_string(b"none") + ssh_string(b"none") + ssh_string(b"")
        + struct.pack(">I", 1)                     # number of keys (bare uint32)
        + ssh_string(blob)
        + ssh_string(priv_body)
    )
    priv_b64 = base64.b64encode(inner).decode()
    pem = ("-----BEGIN OPENSSH PRIVATE KEY-----\n"
           + "\n".join(priv_b64[i:i + 70] for i in range(0, len(priv_b64), 70))
           + "\n-----END OPENSSH PRIVATE KEY-----\n")

    os.makedirs(outdir, mode=0o700, exist_ok=True)
    priv_path = os.path.join(outdir, "id_ed25519_vanity")
    pub_path = os.path.join(outdir, "id_ed25519_vanity.pub")
    with open(priv_path, "w") as f:
        f.write(pem)
    with open(pub_path, "w") as f:
        f.write(f"ssh-ed25519 {b64} {comment}\n".rstrip() + "\n")
    os.chmod(priv_path, 0o600)
    os.chmod(pub_path, 0o644)

    # Independent verification via ssh-keygen.
    derived = subprocess.run(["ssh-keygen", "-y", "-f", priv_path],
                             capture_output=True, text=True, check=True).stdout.strip()
    expected = f"ssh-ed25519 {b64}"
    if derived != expected:
        raise SystemExit("ERROR: ssh-keygen -y does not match derived public key")
    print("ssh-keygen -y verification: OK")
    print("public key line:")
    print(open(pub_path).read(), end="")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise SystemExit(__doc__)
    seed_to_keypair(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "vanity_keys")
