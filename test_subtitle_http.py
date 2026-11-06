"""Only a local fake provider is contacted; no user API key is read."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import subtitle_translator as vt


class ProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                cls.calls.append((self.path, self.headers.get("Authorization") == "Bearer fake-key"))
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/redirect':
                    self.send_response(307)
                    self.send_header('Location', '/must-not-receive-key')
                    self.end_headers()
                    return
                if self.path == '/unauthorized':
                    self.send_response(401)
                    self.end_headers()
                    self.wfile.write(b'private-provider-error-body')
                    return
                if self.path == '/retry' and sum(path == '/retry' for path, _ in cls.calls) == 1:
                    self.send_response(429)
                    self.send_header('Retry-After', '0')
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                if self.path == '/truncated':
                    self.send_header('Content-Length', '100000')
                elif self.path == '/oversized':
                    self.send_header('Content-Length', str(3 * 1024 * 1024))
                self.end_headers()
                if self.path == '/invalid':
                    self.wfile.write(b'not-json')
                else:
                    self.wfile.write(json.dumps({'choices': [{'message': {'content': '["你好"]'}}]}).encode())

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def setUp(self):
        self.calls.clear()

    def settings(self, path):
        return vt.Settings(key='fake-key', url=self.base + path, timeout=2, batch_delay=0)

    def test_actual_http_request_and_response(self):
        result = vt.chat(self.settings('/ok'), [{'role': 'user', 'content': '["Hello"]'}])
        self.assertEqual(result, '["你好"]')
        self.assertEqual(self.calls, [('/ok', True)])

    def test_auth_failure_hides_upstream_body(self):
        with self.assertRaises(vt.ApiError) as caught:
            vt.chat(self.settings('/unauthorized'), [])
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn('private-provider-error-body', str(caught.exception))
        self.assertNotIn('fake-key', str(caught.exception))

    def test_redirect_never_forwards_authorization(self):
        with self.assertRaises(vt.ApiError):
            vt.chat(self.settings('/redirect'), [])
        self.assertEqual(self.calls, [('/redirect', True)])

    def test_actual_rate_limit_recovers(self):
        sleep = Mock()
        result = vt.api_batch(['Hello'], 'en', self.settings('/retry'), sleep=sleep)
        self.assertEqual(result, ['你好'])
        self.assertEqual(len(self.calls), 2)
        sleep.assert_called_once_with(1)

    def test_invalid_provider_json_is_reported(self):
        with self.assertRaises(vt.ApiError) as caught:
            vt.chat(self.settings('/invalid'), [])
        self.assertTrue(caught.exception.retryable)

    def test_truncated_valid_json_is_not_accepted(self):
        with self.assertRaises(vt.ApiError) as caught:
            vt.chat(self.settings('/truncated'), [])
        self.assertTrue(caught.exception.retryable)

    def test_oversized_response_is_rejected(self):
        with self.assertRaises(vt.ApiError):
            vt.chat(self.settings('/oversized'), [])


if __name__ == '__main__':
    unittest.main()
