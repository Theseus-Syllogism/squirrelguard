// Capture redacted screenshots of the persons UI and the 3D replay demo for the public docs (img/screens/).
// Run against an SSH tunnel to the loopback-only persons UI (127.0.0.1:18092) as the server's administrator:
//   SIGHTING=<id> IDENTITY=<id> REDACT_FILE=redact.local.json CHROME=/path/to/chrome node capture_screens.cjs
// Redaction is done in the page: all person/face images blurred, names, camera names, heights and form values
// replaced from REDACT_FILE (untracked: [["regex","flags","replacement"], ...]), third parties' columns blurred.
// LOOK AT EVERY IMAGE before committing: new page content can need new rules.
const fs = require("fs");
const OUT = process.env.OUT || "shots/", BASE = "http://127.0.0.1:18092";
const RULES = JSON.parse(fs.readFileSync(process.env.REDACT_FILE || "redact.local.json", "utf8"));
const CSS = 'main img, img[src*="/crop/"], img[src*="face"], img[src*="thumbnail"], svg image, video { filter: blur(16px) saturate(.6) !important; }';
async function redact(page, names) {
  await page.addStyleTag({ content: CSS });
  await page.evaluate(([names, RULES]) => {
    const map = RULES.map(([re, f, to]) => [new RegExp(re, f), to]);
    names.forEach((n, i) => map.push([new RegExp("\\b" + n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\b", "g"), "Person " + "ABCDEFGH"[i % 8]]));
    const fx = (t) => { for (const [re, s] of map) t = t.replace(re, s); return t; };
    const fix = (root) => { const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT); let n;
      while ((n = w.nextNode())) { const t = fx(n.nodeValue); if (t !== n.nodeValue) n.nodeValue = t; }
      root.querySelectorAll?.("[title]").forEach((e) => { e.title = fx(e.title); }); };
    fix(document.body);
    document.querySelectorAll('input[name="label"]').forEach((i) => { if (i.value) i.value = "Person A"; });
    document.querySelectorAll('input[name="notes"]').forEach((i) => { if (i.value) i.value = "(notes redacted)"; });
    document.querySelectorAll("option").forEach((o) => { o.textContent = fx(o.textContent); });
    new MutationObserver((ms) => ms.forEach((m) => { if (m.type === "characterData") { const t = fx(m.target.nodeValue); if (t !== m.target.nodeValue) m.target.nodeValue = t; }
      m.addedNodes.forEach((x) => x.nodeType === 1 ? fix(x) : x.nodeType === 3 ? (x.nodeValue = fx(x.nodeValue)) : null); }))
      .observe(document.body, { childList: true, subtree: true, characterData: true });
  }, [names, RULES]);
}
(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const b = await chromium.launch({ executablePath: process.env.CHROME,
    args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const mk = async (o) => { const c = await b.newContext(o); await c.route(BASE + "/**", (r) => r.continue({ headers: { ...r.request().headers(), "remote-user": "docs" } })); return c; };
  const ctx = await mk({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1.5, colorScheme: "light" });
  const p = await ctx.newPage();
  await p.goto(BASE + "/"); await p.waitForTimeout(1200);
  const names = await p.evaluate(() => [...new Set([...document.querySelectorAll('a[href*="/identity/"]')].map((a) => a.textContent.trim())
     .filter((t) => t && !/^unknown/i.test(t) && !/^#?\d+$/.test(t) && t.length < 40))]);
  const shot = async (pg, n) => { await pg.waitForTimeout(700); await pg.screenshot({ path: OUT + n + ".png" }); console.log("shot", n); };
  const cols = "table tr td:nth-child(4), table tr td:nth-child(6), table tr td:nth-child(7) { filter: blur(5px); }";
  await redact(p, names); await p.addStyleTag({ content: cols }); await shot(p, "persons-index");
  await p.goto(BASE + "/identity/" + (process.env.IDENTITY || "1") + "?done=saved"); await redact(p, names); await shot(p, "persons-identity");
  await p.goto(BASE + "/identity/" + (process.env.IDENTITY || "1") + ""); await p.waitForTimeout(800); await redact(p, names);
  await p.selectOption('select[name="into"]', { index: 1 }); await redact(p, names);
  await p.click("button.warn"); await p.waitForTimeout(700); await redact(p, names); await shot(p, "persons-merge-dialog");
  await p.keyboard.press("Escape");
  await p.goto(BASE + "/sighting/" + process.env.SIGHTING); await p.waitForTimeout(1500); await redact(p, names);
  await p.addStyleTag({ content: "canvas { filter: blur(14px) !important; }" });
  await p.evaluate(() => document.querySelectorAll("table").forEach((t) => { if (/IPD|feature|Uncertainty/i.test(t.textContent)) t.style.filter = "blur(5px)"; }));
  await shot(p, "persons-why");
  await p.goto(BASE + "/alerts"); await p.waitForTimeout(800); await redact(p, names); await shot(p, "persons-alerts");
  const DEMO = BASE + "/replay/?data=demo/demo_body3d.json&scene=projection&cutouts=trail&body=both";
  await p.goto(DEMO); await p.waitForTimeout(12000); await shot(p, "replay-demo");
  await p.goto(DEMO + "&pane=1&pane_tab=scene"); await p.waitForTimeout(12000); await shot(p, "replay-demo-pane");
  const m = await (await mk({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true })).newPage();
  await m.goto(BASE + "/"); await m.waitForTimeout(1500); await redact(m, names);
  await m.addStyleTag({ content: '.stack td[data-label="Seen"], .stack td[data-label="Height"], .stack td[data-label="Typical clothing"] { filter: blur(5px); }' });
  await shot(m, "persons-mobile");
  await b.close();
})();
