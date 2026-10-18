"""Browser smoke check. All test API responses stay inside Playwright routing."""
import base64
import io
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'service' / 'ui-qa'
OUT.mkdir(exist_ok=True)
CSP = "default-src 'self'; img-src 'self' data:; media-src blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'service'), **kwargs)
    def do_GET(self):
        if self.path.split('?')[0] == '/': self.path = '/static/index.html'
        super().do_GET()
    def end_headers(self):
        self.send_header('Content-Security-Policy', CSP)
        super().end_headers()
    def log_message(self, *args): pass

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f'http://127.0.0.1:{server.server_port}'
buffer = io.BytesIO()
Image.new('RGB', (640, 360), '#d8d6d3').save(buffer, 'JPEG')
evidence = base64.b64encode(buffer.getvalue()).decode()
job = 'a' * 32
fixture = {'job_id': job, 'status': 'done', 'verdict': 'violation', 'violations': [
    {'type': 'Рабочее место оставлено без присмотра', 'timecode_sec': 12.5,
     'description': 'Что видно: Сотрудник покидает рабочее место.\nПочему это важно: Нужно проверить порядок завершения операции.\nЧто проверить: Уточните, закрыта ли касса перед уходом.\nНорма: Тестовый текст источника https://adilet.zan.kz/rus/docs/example',
     'evidence_frames': [{'timestamp_sec': 12.5, 'image_base64': evidence}]},
    {'type': '<img src=x onerror=alert(1)>', 'timecode_sec': 20,
     'description': 'Что видно: <script>alert(1)</script>\nЧто проверить: Тест безопасного отображения текста.',
     'evidence_frames': [{'timestamp_sec': 20, 'image_base64': evidence}]}
]}
checks = []
def check(label): checks.append(label)
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
        errors = []
        page.on('pageerror', lambda err: errors.append(str(err)))
        page.goto(url)
        expect(page.locator('#submit')).to_be_disabled()
        page.locator('.quick-help summary').click()
        expect(page.locator('#help-link')).to_be_visible()
        page.locator('.quick-help summary').click()
        expect(page.locator('#help-link')).not_to_be_visible()
        check('On-demand instructions keep the upload screen focused')
        page.screenshot(path=str(OUT / 'desktop.png'), full_page=True)
        page.click('#nav-help')
        page.select_option('#role', 'manager')
        expect(page.locator('#role-action')).to_have_text('Уточните обстоятельства')
        page.click('#nav-help')
        expect(page.locator('#help-title')).to_be_visible()
        page.click('#help-back')
        expect(page.locator('#check-view')).to_be_visible()
        check('Role guidance and help navigation')
        page.locator('#file').set_input_files({'name': 'wrong.txt', 'mimeType': 'text/plain', 'buffer': b'x'})
        expect(page.locator('#error')).to_be_visible()
        expect(page.locator('#submit')).to_be_disabled()
        page.locator('#file').set_input_files({'name': 'empty.mov', 'mimeType': 'video/quicktime', 'buffer': b''})
        expect(page.locator('#submit')).to_be_disabled()
        check('Invalid and empty file validation')
        calls = {'post': 0, 'get': 0}
        def post(route):
            calls['post'] += 1
            route.fulfill(status=202, json={'job_id': job})
        def get(route):
            calls['get'] += 1
            route.fulfill(json={'job_id': job, 'status': 'processing'} if calls['get'] == 1 else fixture)
        page.route('**/ui/analyze', post)
        page.route('**/ui/jobs/*', get)
        page.locator('#file').set_input_files({'name': 'Касса-01.mov', 'mimeType': 'video/quicktime', 'buffer': b'test video'})
        expect(page.locator('#submit')).to_be_enabled()
        page.locator('#file').set_input_files({'name': 'wrong.txt', 'mimeType': 'text/plain', 'buffer': b'x'})
        expect(page.locator('#submit')).to_be_disabled()
        page.evaluate("""() => {
            const data = new DataTransfer();
            data.items.add(new File(['test video'], 'Касса-01.mov', {type: 'video/quicktime'}));
            document.getElementById('dropzone').dispatchEvent(new DragEvent('drop', {bubbles: true, dataTransfer: data}));
        }""")
        expect(page.locator('#submit')).to_be_enabled()
        check('Invalid replacement cannot submit stale file; drag-and-drop selects a new file')
        page.click('#submit')
        expect(page.locator('#processing-panel')).to_be_visible()
        expect(page.locator('.violation')).to_have_count(2, timeout=10000)
        assert calls['post'] == 1
        expect(page.locator('#codec-note')).to_be_visible()
        expect(page.locator('.timecode').first).to_be_disabled()
        expect(page.locator('.event-title img')).to_have_count(0)
        expect(page.locator('.description script')).to_have_count(0)
        check('Upload, processing, result, codec fallback and escaped model text')
        page.locator('.review-checkbox input').first.check()
        expect(page.locator('#review-count')).to_have_text('Просмотрено 1 из 2')
        page.locator('details.norm summary').click()
        expect(page.locator('details.norm a')).to_have_attribute('href', 'https://adilet.zan.kz/rus/docs/example')
        page.locator('.frame button').first.click()
        expect(page.locator('.frame').first).to_have_class('frame expanded')
        page.screenshot(path=str(OUT / 'result.png'), full_page=True)
        check('Reviewed state, source disclosure and frame expansion')
        page.emulate_media(media='print')
        expect(page.locator('.site-header')).not_to_be_visible()
        expect(page.locator('.violation').first).to_be_visible()
        expect(page.locator('.timecode').first).to_be_visible()
        page.emulate_media(media='screen')
        check('Print layout retains events and timestamps')
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(OUT / 'result-mobile.png'), full_page=True)
        page.click('#new-check')
        expect(page.locator('#submit')).to_be_disabled()
        expect(page.locator('#nav-result')).to_be_disabled()
        assert '?' not in page.url
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(OUT / 'mobile.png'), full_page=True)
        check('Mobile layout and clean reset')
        page.goto(url + '/?job=' + job)
        expect(page.locator('.violation')).to_have_count(2)
        expect(page.locator('#empty-preview')).to_be_visible()
        expect(page.locator('.timecode').first).to_be_disabled()
        check('Direct result link without source video')
        page.unroute('**/ui/jobs/*')
        page.route('**/ui/jobs/*', lambda route: route.fulfill(json={'job_id': job, 'status': 'done', 'verdict': 'no_violation', 'violations': []}))
        page.reload()
        expect(page.locator('#clear-result')).to_be_visible()
        expect(page.locator('#verdict')).to_have_text('Нарушение не обнаружено')
        check('No-violation result')
        page.unroute('**/ui/jobs/*')
        page.route('**/ui/jobs/*', lambda route: route.fulfill(status=404, json={'detail': 'Job not found'}))
        page.reload()
        expect(page.locator('#result-error')).to_be_visible()
        expect(page.locator('#result-retry')).not_to_be_visible()
        expect(page.locator('#new-check')).to_be_enabled()
        check('Missing result with recovery action')
        page.unroute('**/ui/jobs/*')
        page.route('**/ui/jobs/*', lambda route: route.fulfill(json={'job_id': job, 'status': 'failed', 'error': 'Processing time limit exceeded'}))
        page.reload()
        expect(page.locator('#result-error-text')).to_contain_text('более короткий')
        check('Readable processing failure')
        page.unroute('**/ui/jobs/*')
        page.route('**/ui/jobs/*', lambda route: route.fulfill(status=503, json={'detail': 'unavailable'}))
        page.reload()
        expect(page.locator('#result-retry')).to_be_visible(timeout=15000)
        page.unroute('**/ui/jobs/*')
        page.route('**/ui/jobs/*', lambda route: route.fulfill(json=fixture))
        page.click('#result-retry')
        expect(page.locator('.violation')).to_have_count(2)
        assert calls['post'] == 1
        check('Interrupted polling can resume without uploading again')
        assert not errors, errors
        check('No JavaScript runtime errors')
        browser.close()
finally:
    server.shutdown()
print(json.dumps({'passed': len(checks), 'checks': checks}, ensure_ascii=True, indent=2))
