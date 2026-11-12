"""Checks for the local-only dashboard simulation API."""
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('sim_server', Path(__file__).resolve().parents[1] / 'dashboard-web' / 'sim_server.py')
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)

@pytest.fixture
def api():
    server = HTTPServer(('127.0.0.1', 0), server_module._Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    thread.join()

def request(api, body, origin='http://localhost:5173'):
    req = urllib.request.Request(api + '/simulate', data=body, headers={'Content-Type': 'application/json', 'Origin': origin})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.load(response), response.headers
    except urllib.error.HTTPError as error:
        return error.code, json.load(error), error.headers

def test_valid_small_simulation(api):
    status, data, headers = request(api, b'{"protocol":"naive","nodes":10,"degree":2,"runs":1}')
    assert status == 200 and 'summary' in data
    assert headers['Access-Control-Allow-Origin'] == 'http://localhost:5173'

def test_rejects_untrusted_origin_and_excessive_input(api):
    assert request(api, b'{}', 'https://example.com')[0] == 403
    assert request(api, b'{"runs":100000}')[0] == 400
    assert request(api, b'[')[0] == 400
    assert request(api, b' ' * 16385)[0] == 413

