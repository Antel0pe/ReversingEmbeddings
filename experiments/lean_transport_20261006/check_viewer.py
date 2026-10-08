"""Check the saved HTML in an available local headless Chromium browser."""
from pathlib import Path
import html
import json
import os
import re
import shutil
import subprocess
import tempfile

OUT=Path(__file__).resolve().parent/'results'

def main():
    supplied=os.environ.get('LEAN_BROWSER_BIN')
    choices=list((Path.home()/'.cache/ms-playwright').glob('chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell'))
    browser=supplied or shutil.which('chromium') or shutil.which('google-chrome') or (str(sorted(choices)[-1]) if choices else None)
    if not browser:raise RuntimeError('Set LEAN_BROWSER_BIN to an available local Chromium executable')
    reports=[]
    with tempfile.TemporaryDirectory(prefix='lean-viewer-') as temp:
        for w,h,label in [(1200,1050,'desktop'),(390,2200,'mobile')]:
            result=subprocess.run([browser,'--no-sandbox','--disable-gpu',f'--window-size={w},{h}',
                f'--user-data-dir={temp}/{label}',f'--screenshot={OUT}/viewer_{label}.png','--dump-dom',
                (OUT/'viewer.html').as_uri()+'?selftest=1'],capture_output=True,text=True,timeout=60)
            match=re.search(r'<pre id="selftest-report">(.*?)</pre>',result.stdout)
            if result.returncode or not match:raise RuntimeError(f'{label}: browser check did not complete')
            report=json.loads(html.unescape(match.group(1)))
            assert report['passed'] and report['renderedStates']==1008
            assert report['canvasCount']==4 and report['initialIndex']==58
            assert not report['horizontalOverflow']
            reports.append({'viewport_width':w,'viewport_height':h,**report})
    (OUT/'browser_checks.json').write_text(json.dumps({'browser':'local headless Chromium','checks':reports},indent=2)+'\n')
    print(json.dumps(reports))

if __name__=='__main__':main()
