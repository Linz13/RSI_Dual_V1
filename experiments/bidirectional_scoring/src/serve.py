"""Loopback annotation server: serves only blinded assets, never model results/manifests."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from common import load_manifest, annotations, save_annotations, within, shared_runtime


def make_handler(run, m):
    review = (run / "review").resolve()
    allowed_audio = {r["audio"] for r in m["identity"]["a"]}
    allowed_audio.update(c["audio"] for r in m["identity"]["b"] for c in r["candidates"])

    class Handler(BaseHTTPRequestHandler):
        def json_response(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            route = unquote(urlsplit(self.path).path).lstrip("/")
            if route == "api/annotations":
                self.json_response(annotations(run, m))
                return
            if route in ("", "index.html", "style.css", "app.js", "data.js"):
                path = review / (route or "index.html")
            elif route == "bundle.zip":
                path = run / "annotation_bundle.zip"
            elif route in allowed_audio:
                path = review / route
                if not within(path, review / "audio"):
                    self.send_error(403)
                    return
            else:
                self.send_error(404)
                return
            if not path.is_file():
                self.send_error(404)
                return
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), end) if right else end
                else:
                    start = max(0, size - int(right))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = 206
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(end-start+1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if route == "bundle.zip":
                self.send_header("Content-Disposition", 'attachment; filename="annotation_bundle.zip"')
            self.end_headers()
            if self.command != "HEAD":
                try:
                    with path.open("rb") as f:
                        f.seek(start)
                        remaining = end-start+1
                        while remaining:
                            chunk = f.read(min(256 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_POST(self):
            if self.path != "/api/annotations":
                self.send_error(404)
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.json_response({"error": "只接受同一页面的保存请求"}, 403)
                return
            try:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("需要 JSON 请求")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2 * 1024 * 1024:
                    raise ValueError("请求大小不正确")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body.get("expected_revision"), int):
                    raise ValueError("缺少保存版本")
                result = save_annotations(run, m, body["annotations"], body["expected_revision"])
                self.json_response(result)
            except (ValueError, KeyError, TypeError, RuntimeError) as e:
                self.json_response({"error": str(e)}, 409)

    return Handler


def main():
    shared_runtime()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    run, m = load_manifest(args.run)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(run, m))
    print(f"标注页面：http://127.0.0.1:{args.port}", flush=True)
    print(f"标注保存：{run / 'annotations.json'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
