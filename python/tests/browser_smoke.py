"""Optional real-browser check. Start app.py --no-browser on port 8765 first."""

from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto("http://127.0.0.1:8765")
        page.wait_for_function("document.querySelector('#summary').children.length>0")
        page.wait_for_function(
            "document.querySelector('#plotInfo').textContent.includes('önizleme')"
        )
        assert page.locator("canvas").count() == 3
        page.select_option("#flight", "1")
        page.click("#plot")
        page.fill("#from", "2")
        page.fill("#to", "20")
        page.click("#zoom")
        page.select_option("#source", "user")
        uploads = []
        for j in range(4):
            t = np.arange(512) / 64
            u = 0.1 * np.sin(t * (1 + j * 0.1))
            z = 0.8 * u
            x = np.cumsum(z) / 64
            rows = ["time,CoilCurrent,DDV,RAM,Deflection"] + [
                ",".join(map(str, r)) for r in zip(t, u, z, x, 0.3 * x)
            ]
            uploads.append(
                {
                    "name": f"check_{j}.csv",
                    "mimeType": "text/csv",
                    "buffer": "\n".join(rows).encode(),
                }
            )
        page.set_input_files("#files", uploads)
        page.click("#inspect")
        page.wait_for_selector("#signalCoilCurrent")
        page.fill("#assignments", "train,train,validation,test")
        page.click("#import")
        page.wait_for_function(
            "document.querySelector('#importStatus').textContent.includes('4 uçuş')"
        )
        page.select_option("#methods", "oe")
        page.check("#test")
        page.click("#run")
        page.wait_for_function(
            "document.querySelector('#status').textContent==='Tamamlandı.'",
            timeout=90000,
        )
        page.wait_for_function(
            "document.querySelector('#sourceName').textContent.startsWith('user')"
        )
        assert page.locator("#summary tr").count() >= 1
        href = page.locator("#xlsx").get_attribute("href")
        response = page.request.get("http://127.0.0.1:8765" + href)
        assert response.ok and response.body()[:2] == b"PK"
        assert not errors, errors
        # Show the shipped pulse experiment in the final screenshot.
        records = page.request.get("http://127.0.0.1:8765/api/results").json()
        ref = next(
            r
            for r in records
            if r["config"]["source"] == "pulse" and r["id"] == "20261112_053356_805234"
        )
        page.select_option("#runs", ref["id"])
        page.wait_for_function(
            "document.querySelector('#sourceName').textContent.startsWith('pulse')"
        )
        page.wait_for_timeout(1000)
        page.screenshot(path=str(Path("/tmp/actuator_python_ui.png")), full_page=True)
        browser.close()
        print(
            "PASS: CSV import, explicit split, OE training, 3 plots, zoom, XLSX download; no JS errors"
        )


if __name__ == "__main__":
    main()
