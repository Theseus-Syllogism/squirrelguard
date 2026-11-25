// Called by capture_console.py: node capture_console.cjs START_URL OUT_DIR SWAPS_JSON [EVIDENCE_PAGES_DIR]
const { chromium } = require('playwright-core');
const path = require('path');
const fs = require('fs');
const [start, out, swapsJson, pages] = process.argv.slice(2);
const swaps = JSON.parse(swapsJson);

// Replace text in every text node and input (addresses, folders) before a capture.
async function swap(page, map) {
  await page.evaluate((map) => {
    const fix = (s) => Object.entries(map).reduce((t, [a, b]) => t.split(a).join(b), s);
    const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) { const f = fix(n.nodeValue); if (f !== n.nodeValue) n.nodeValue = f; }
    document.querySelectorAll('input').forEach((i) => { i.value = fix(i.value); if (i.placeholder) i.placeholder = fix(i.placeholder); });
  }, map);
}

// Narrow viewports on purpose: the pictures are printed about 7 inches wide, and interface text must stay readable.
const CON_W = 900, EV_W = 1000;

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME || undefined });
  const ctx = await browser.newContext({ viewport: { width: CON_W, height: 700 }, deviceScaleFactor: 2, colorScheme: 'light' });
  const page = await ctx.newPage();
  const light = () => page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
  const shot = async (name, opt = {}) => { await page.screenshot({ path: path.join(out, name), ...opt }); console.log('wrote', name); };

  await page.goto(start);
  const origin = new URL(start).origin;
  for (const [hash, name, maxH, after] of [
    ['install', 'console-install.png', 720],
    ['endpoints', 'console-endpoints.png', 0, async () => { await page.click('text=Test all'); await page.waitForSelector('.ep-result >> nth=1'); }],
    ['checks', 'console-checks.png', 700]]) {
    await page.goto(origin + '/#' + hash);
    await page.waitForSelector('main h2');
    await light();
    if (after) await after();
    await page.waitForTimeout(400);
    await swap(page, swaps);
    await page.setViewportSize({ width: CON_W, height: 200 });          // so the page reports its own height
    const h = await page.evaluate(() => document.querySelector('main').getBoundingClientRect().bottom + 20);
    await page.setViewportSize({ width: CON_W, height: Math.round(maxH ? Math.min(h, maxH) : h) });
    await shot(name);
  }

  if (pages && fs.existsSync(pages)) {
    const ev = { 'http://127.0.0.1': 'https://timestamp.example', '127.0.0.1': '192.0.2.21' };
    const open = async (file) => {
      const f = path.join(pages, file + '.html');
      if (!fs.existsSync(f)) { console.log('missing', f); return 0; }
      await page.setViewportSize({ width: EV_W, height: 400 });
      await page.goto('file://' + f);
      await light();
      await page.evaluate(() => document.querySelectorAll('img').forEach((i) => { if (!i.complete || !i.naturalWidth) i.remove(); }));
      await swap(page, ev);
      // the stand-in time service has a port; drop it from the example address
      await page.evaluate(() => { const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); for (let n = w.nextNode(); n; n = w.nextNode()) n.nodeValue = n.nodeValue.replace(/timestamp\.example:\d+\/?/g, 'timestamp.example'); });
      return page.evaluate(() => document.documentElement.scrollHeight);
    };
    // y position of the first heading with this text
    const yOf = (text) => page.evaluate((text) => { const h = [...document.querySelectorAll('h1,h2,h3')].find((e) => e.textContent.trim().startsWith(text)); return h ? h.getBoundingClientRect().top + scrollY : 0; }, text);

    let h = await open('sighting');
    if (h) {
      const walk = await yOf('Walk');
      await page.setViewportSize({ width: EV_W, height: h });
      await shot('evidence-review.png', { clip: { x: 0, y: 0, width: EV_W, height: Math.round(walk - 14) } });
      const end = await page.evaluate((walk) => { const s = [...document.querySelectorAll('svg')].filter((e) => e.getBoundingClientRect().top + scrollY > walk).pop();
        return s ? s.getBoundingClientRect().bottom + scrollY + 14 : document.documentElement.scrollHeight; }, walk);
      await shot('gait-walk-comparison.png', { clip: { x: 0, y: Math.round(walk - 14), width: EV_W, height: Math.round(end - walk + 14) } });
    }
    for (const [file, name, maxH] of [['export-form', 'evidence-export-form.png', 0], ['export', 'evidence-package.png', 0], ['audit', 'evidence-audit.png', 640]]) {
      h = await open(file);
      if (!h) continue;
      await page.setViewportSize({ width: EV_W, height: 200 });
      const bottom = await page.evaluate(() => Math.max(...[...document.querySelectorAll('main > *, body > main *')].map((e) => e.getBoundingClientRect().bottom)) + 18);
      await page.setViewportSize({ width: EV_W, height: Math.round(maxH ? Math.min(bottom, maxH) : bottom) });
      await shot(name);
    }
  }
  await browser.close();
})();
