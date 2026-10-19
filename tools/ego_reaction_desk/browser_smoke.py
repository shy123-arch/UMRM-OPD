"""Browser acceptance against synthetic media and disposable data only."""
import threading
from playwright.sync_api import sync_playwright, expect
from werkzeug.serving import make_server
from test_desk import DeskTests
from server import create_app


def main():
    DeskTests.setUpClass(); fixture=DeskTests();fixture.setUp()
    server=make_server('127.0.0.1',0,create_app(fixture.store),threaded=True)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1440,'height':1100},accept_downloads=True)
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}/')
            expect(page.locator('h1')).to_have_text('ego数据反应标注台')
            expect(page.locator('#reaction-cn')).to_have_value('拿起盒子。')
            page.wait_for_function('document.getElementById("video").readyState >= 2')
            page.locator('#actor').fill('QA')
            page.locator('#start').fill('0.2');page.locator('#end').fill('2.2')
            page.locator('#note').fill('可恢复草稿')
            page.once('dialog',lambda d:d.accept());page.reload()
            expect(page.locator('#note')).to_have_value('可恢复草稿')
            page.locator('#actor').fill('QA')
            page.locator('#boundary').click()
            expect(page.locator('#frames')).to_be_visible(timeout=15000)
            page.locator('#boundary-ok').check();page.locator('#support-ok').check()
            page.locator('#save').click()
            expect(page.locator('#save-state')).to_contain_text('已保存',timeout=10000)
            assert fixture.store.get('fixture_1')['status']=='queued'
            fixture.store.process_one()
            expect(page.locator('#result')).to_contain_text('播放已生成切片',timeout=12000)
            with page.expect_download() as info:page.locator('#export').click()
            assert info.value.suggested_filename.endswith('.zip')
            page.locator('#result button').click()
            expect(page.locator('#video-mode')).to_contain_text('反应前切片')
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert not errors,errors
            browser.close()
        print('PASS browser: prefill, playback, draft recovery, original frames, save/cut, download, mobile layout')
    finally:
        server.shutdown();fixture.tearDown();DeskTests.tearDownClass()


if __name__=='__main__':main()
