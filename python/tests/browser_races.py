"""Actual Chromium regression against a saved result; run app on port 8771."""

import asyncio
import json
from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"]
        )
        page = await browser.new_page(viewport=dict(width=1280, height=950))
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto("http://127.0.0.1:8771")
        await page.wait_for_function(
            "document.querySelector('#plotInfo').textContent.includes('önizleme')",
            timeout=180000,
        )
        assert await page.locator("canvas").count() == 3
        assert await page.evaluate("document.querySelector('#chart0').width>0")
        # Delay the first requested flight response; a newer request must win.
        original = await page.evaluate("plotData")

        async def delayed(route):
            from urllib.parse import urlparse, parse_qs

            flight = int(parse_qs(urlparse(route.request.url).query)["flight"][0])
            await asyncio.sleep(0.7 if flight == 0 else 0.05)
            data = dict(original, samples=10000 + flight)
            await route.fulfill(json=data)

        await page.route("**/plot?*", delayed)
        await page.select_option("#flight", "0")
        await page.click("#plot")
        await page.select_option("#flight", "1")
        await page.wait_for_function("plotData && plotData.samples===10001")
        await page.wait_for_timeout(1000)
        assert await page.evaluate("plotData.samples") == 10001
        # Loading another result invalidates an older in-flight response too.
        await page.evaluate("$('flight').value='0';plot();load(result.id)")
        await page.wait_for_function(
            "document.querySelector('#plotInfo').textContent.includes('önizleme')"
        )
        # Visible failures and subsequent retry.
        await page.unroute("**/plot?*", delayed)

        async def failure(route):
            await route.fulfill(
                status=500, json={"detail": "intentional regression failure"}
            )

        await page.route("**/plot?*", failure)
        await page.click("#plot")
        await page.wait_for_function(
            "document.querySelector('#error').textContent.includes('intentional regression failure')"
        )
        assert await page.locator("#error").is_visible()
        await page.unroute("**/plot?*", failure)
        await page.click("#plot")
        await page.wait_for_function(
            "document.querySelector('#plotInfo').textContent.includes('önizleme')",
            timeout=180000,
        )
        # Real multipart upload + selected-column import through the UI.
        await page.select_option("#source", "user")
        rows = ["time,CoilCurrent,DDV,RAM,Deflection"] + [
            f"{i / 64},0.1,0.1,{i / 640}, {-i / 640}" for i in range(2048)
        ]
        await page.set_input_files(
            "#files",
            [
                dict(
                    name="browser.csv",
                    mimeType="text/csv",
                    buffer="\n".join(rows).encode(),
                )
            ],
        )
        await page.click("#inspect")
        await page.wait_for_selector("#signalCoilCurrent")
        await page.click("#import")
        await page.wait_for_function(
            "document.querySelector('#importStatus').textContent.includes('1 uçuş hazır')"
        )
        assert not errors, errors
        print(
            json.dumps(
                dict(
                    saved_result=True,
                    first_open_plots=3,
                    stale_flight_ignored=True,
                    load_race=True,
                    visible_error_retry=True,
                    multipart_import=True,
                    page_errors=errors,
                )
            )
        )
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
