#!/usr/bin/env python3
"""Generate the public architecture diagrams (draw.io) for squirrel-guard.

    python3 docs/public/tools/make_diagrams.py          # writes docs/public/diagrams/squirrel-guard-architecture.drawio
    python3 docs/public/tools/make_diagrams.py --export # also PNG + SVG per page (needs the drawio CLI)

Public-release rules for everything drawn here: host ROLES only (no hostnames, addresses, domains, accounts or
credentials), open-source component names with the versions pinned in the repo, icons from Lucide (ISC licence,
docs/public/icons). Edit this script, not the .drawio file, so the diagrams stay reproducible.
"""
import base64
import html
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ICONS = ROOT / "icons"
OUT = ROOT / "diagrams"

# palette (WCAG AA text contrast on the fills)
NAVY, INK, GREY = "#1F3B57", "#1B1F24", "#5B6670"
ROLE = {  # host role -> (fill, stroke)
    "camera": ("#EEF1F4", "#5B6670"),
    "recorder": ("#E3EEF9", "#2E6DA4"),
    "gpu": ("#E6F4EA", "#2E7D45"),
    "operator": ("#F1E9F7", "#6B3FA0"),
    "external": ("#FFF4E0", "#B36B00"),
}
ROLE_NAME = {"camera": "Camera network", "recorder": "Recorder host (CPU)", "gpu": "GPU server",
             "operator": "Operator workstation (optional)", "external": "Outside the system"}


def icon_uri(name, color=NAVY):
    svg = (ICONS / f"{name}.svg").read_text().replace('stroke="currentColor"', f'stroke="{color}"')
    return "data:image/svg+xml," + base64.b64encode(svg.encode()).decode()


class Page:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.cells, self.n = name, w, h, [], 1

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def add(self, value, style, x, y, w, h, parent="1"):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{html.escape(value, quote=True)}" style="{style}" vertex="1" '
                          f'parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def text(self, value, x, y, w, h, size=12, bold=False, color=INK, align="left", valign="middle"):
        st = (f"text;html=1;whiteSpace=wrap;fontFamily=Helvetica;fontSize={size};fontColor={color};align={align};"
              f"verticalAlign={valign};{'fontStyle=1;' if bold else ''}")
        return self.add(value, st, x, y, w, h)

    def band(self, title, x, y, w, h, fill="#F7F9FB", stroke="#C9D1D9"):
        self.add("", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};", x, y, w, h)
        self.text(f"<b>{title}</b>", x + 8, y + 4, 150, h - 8, size=12, color=NAVY, valign="middle")

    def box(self, title, sub, x, y, w, h, role="recorder", icon=None, dashed=False):
        fill, stroke = ROLE[role]
        label = f"<b>{html.escape(title)}</b>" + (f"<br><font style=\"font-size:10px\" color=\"{GREY}\">{html.escape(sub)}</font>" if sub else "")
        pad = 34 if icon else 8
        st = (f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
              f"fontFamily=Helvetica;fontSize=12;fontColor={INK};align=left;spacingLeft={pad};spacingRight=6;"
              f"{'dashed=1;' if dashed else ''}")
        b = self.add(label, st, x, y, w, h)
        if icon:
            self.add("", f"shape=image;image={icon_uri(icon, stroke)};imageAspect=1;", x + 7, y + (h - 22) / 2, 22, 22)
        return b

    def edge(self, a, b, label="", dashed=False, color=GREY, exit=None, entry=None, both=False, points=()):
        i = self._id()
        st = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor={color};strokeWidth=1.5;fontSize=10;"
              f"fontColor={INK};fontFamily=Helvetica;labelBackgroundColor=#FFFFFF;endArrow=block;endFill=1;"
              f"{'dashed=1;' if dashed else ''}{'startArrow=block;startFill=1;' if both else ''}")
        if exit:
            st += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        if entry:
            st += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        pts = ('<Array as="points">' + "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points) + "</Array>") if points else ""
        self.cells.append(f'<mxCell id="{i}" value="{html.escape(label, quote=True)}" style="{st}" edge="1" parent="1" '
                          f'source="{a}" target="{b}"><mxGeometry relative="1" as="geometry">' + pts + '</mxGeometry></mxCell>')
        return i

    def legend(self, x, y, roles=("camera", "recorder", "gpu", "operator", "external")):
        self.text("<b>Where it runs</b>", x, y, 200, 18, size=11, color=NAVY)
        for k, r in enumerate(roles):
            fill, stroke = ROLE[r]
            self.add("", f"rounded=1;fillColor={fill};strokeColor={stroke};strokeWidth=1.5;", x, y + 24 + k * 22, 22, 14)
            self.text(ROLE_NAME[r], x + 30, y + 21 + k * 22, 230, 20, size=11)

    def header(self, title, sub):
        self.text(f"<b>{title}</b>", 30, 18, self.w - 60, 30, size=20, color=NAVY)
        self.text(sub, 30, 48, self.w - 60, 22, size=12, color=GREY)
        self.text("squirrel-guard · public architecture · icons: Lucide (ISC)", 30, self.h - 28, self.w - 60, 18,
                  size=9, color=GREY, align="right")

    def xml(self, pid):
        body = "".join(self.cells)
        return (f'<diagram id="{pid}" name="{html.escape(self.name)}"><mxGraphModel dx="1400" dy="900" grid="1" '
                f'gridSize="10" guides="1" page="1" pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" '
                f'background="#FFFFFF" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}'
                f'</root></mxGraphModel></diagram>')


# --------------------------------------------------------------------------------------------- page 1: stack
def page_stack():
    p = Page("1 Software stack", 1654, 1190)
    p.header("Software stack", "Every component of the deployed system, by layer. Colour shows which machine role runs it.")
    X, W, top, rowh, gap = 30, 1360, 86, 92, 8
    rows = [
        ("Cameras", [("IP cameras", "open Thingino firmware · RTSP main + sub stream", "camera", "cctv"),
                     ("USB webcams", "optional test / desk cameras", "operator", "camera"),
                     ("Synthetic test source", "MediaMTX 1.21.3 · injects test animals", "gpu", "film")]),
        ("Stream relay", [("go2rtc 1.9.9", "one connection per camera, re-served over authenticated RTSP", "recorder", "radio")]),
        ("Recording &\ndetection", [("Continuous recorder", "ffmpeg 10-min segments · retention janitor (7 days / size cap)", "recorder", "hard-drive"),
                                    ("Frigate 0.18.2 NVR", "OpenVINO YOLOv8s (Open Images, 601 classes) · VAAPI decode · event clips", "gpu", "video")]),
        ("Message bus", [("Eclipse Mosquitto 2.0.20", "MQTT · password auth · frigate/* and squirrelguard/<camera>/* topics", "recorder", "workflow")]),
        ("Analysis", [("Listener", "verify animal · trigger deterrent · clips", "gpu", "squirrel"),
                      ("Profiler", "people & vehicles: re-ID, gait, pose, face, attributes", "gpu", "user-search"),
                      ("Bridge", "matching, rules, storage, notifications", "recorder", "git-branch"),
                      ("Scene worker", "multi-camera 3D solve, Gaussian splats", "operator", "boxes")]),
        ("Storage", [("PostgreSQL 15", "events, identities, sightings, vehicles, alerts, face templates", "recorder", "database"),
                     ("File storage", "recordings, snapshots, evidence, replay data", "recorder", "hard-drive"),
                     ("Frigate media", "event clips, semantic-search index", "gpu", "hard-drive")]),
        ("AI models", [("Local vision-language model", "llama.cpp server · OpenAI-compatible API · no cloud", "gpu", "brain"),
                       ("ONNX model zoo", "ONNX Runtime 1.20 (CPU): detection, re-ID, gait, pose, face", "gpu", "cpu"),
                       ("TensorRT (optional)", "fp16 engines on an NVIDIA GPU", "operator", "gauge")]),
        ("User interfaces", [("Portal", "status + live view", "recorder", "monitor"),
                             ("Persons UI", "identities, alerts, reports", "recorder", "users"),
                             ("3D replay", "three.js + Gaussian splats", "recorder", "map"),
                             ("Frigate UI", "review, clips", "gpu", "play"),
                             ("Grafana 11.4", "dashboards", "recorder", "chart-line")]),
        ("Access control", [("nginx", "TLS reverse proxy, one host per UI", "recorder", "network"),
                            ("Authelia 4.39", "single sign-on · 2FA · passkeys · default deny", "recorder", "key-round"),
                            ("ACME certificates", "lego 4.35 · DNS-01", "recorder", "lock")]),
        ("Alerts &\nactuators", [("ntfy 2.11", "push notifications · deny-all default", "recorder", "bell"),
                                ("Email", "SMTP relay, rate-limited", "external", "mail"),
                                ("Camera speaker", "deterrent sound via SSH forced command", "camera", "volume-2")]),
        ("Security &\noperations", [("Suricata IDS/IPS", "inline + passive mirror", "gpu", "shield"),
                                   ("Uptime Kuma 1.23", "service monitors", "recorder", "activity"),
                                   ("Stall guard", "restarts a hung recorder", "gpu", "siren"),
                                   ("Uplink failover", "backup gateway", "recorder", "route")]),
    ]
    ids = {}
    for r, (title, comps) in enumerate(rows):
        y = top + r * (rowh + gap)
        p.band(title.replace("\n", "<br>"), X, y, W, rowh)
        n = len(comps)
        cw = (W - 170 - 12 * (n - 1) - 12) / n
        for k, (t, s, role, ic) in enumerate(comps):
            ids[t] = p.box(t, s, X + 170 + k * (cw + 12), y + 12, cw, rowh - 24, role, ic)
    p.legend(1420, 96)
    p.text("<b>Design rules</b><br>• All AI runs locally; no cloud services.<br>• Only the recorder host holds camera "
           "credentials.<br>• Every web UI is behind single sign-on.<br>• Optional parts are labelled; the system runs "
           "without the operator workstation.", 1420, 250, 210, 170, size=11, valign="top")
    return p


# --------------------------------------------------------------------------------------------- page 2: event pipeline
def page_pipeline():
    p = Page("2 Event pipeline", 1654, 1000)
    p.header("From camera frame to alert", "One detection event, step by step (animal path on top, people and vehicles below).")
    y1, h = 110, 92
    steps = [("Camera", "streams H.264 over RTSP", "camera", "cctv"),
             ("go2rtc relay", "one camera connection, shared", "recorder", "radio"),
             ("Frigate", "motion → YOLOv8s → tracking → event + clip", "gpu", "video"),
             ("MQTT", "frigate/events · frigate/reviews", "recorder", "workflow"),
             ("Listener", "fetch snapshot, filter, de-duplicate", "gpu", "squirrel"),
             ("Vision check", "local VLM answers one word: squirrel / cat / …", "gpu", "brain")]
    ids, x, w = [], 30, 245
    for k, (t, s, role, ic) in enumerate(steps):
        ids.append(p.box(f"{k + 1}. {t}", s, x + k * (w + 22), y1, w, h, role, ic))
    for a, b in zip(ids, ids[1:]):
        p.edge(a, b)
    y2 = 300
    deter = p.box("7. Deterrent", "quiet hours + 60 s cooldown → sound on the camera speaker", 30 + 5 * (w + 22), y2, w, h, "camera", "volume-2")
    pub = p.box("8. Publish", "squirrelguard/<cam>/event + boxed clip", 30 + 4 * (w + 22), y2, w, h, "gpu", "film")
    bridge = p.box("9. Bridge", "database row, files, rules", 30 + 3 * (w + 22), y2, w, h, "recorder", "git-branch")
    notify = p.box("10. Notify", "push with snapshot, then clip · email", 30 + 2 * (w + 22), y2, w, h, "recorder", "bell")
    ui = p.box("11. Review", "SSO → Frigate review, persons UI, dashboards", 30 + 1 * (w + 22), y2, w, h, "recorder", "monitor")
    p.edge(ids[5], deter, "confirmed")
    p.edge(deter, pub)
    p.edge(pub, bridge)
    p.edge(bridge, notify)
    p.edge(notify, ui)
    no = p.box("Not confirmed", "logged, no sound, no alert (detector alone is not trusted)", 30, y2, w, h, "external", "triangle-alert", dashed=True)
    p.edge(ids[5], no, "other answer", dashed=True, exit=(0.5, 1), entry=(0.5, 0))

    # people / vehicles lane (starts from step 4; a note instead of a long crossing edge)
    y3 = 470
    p.add("", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F4FAF5;strokeColor=#B9D9C0;", 30, y3, 1594, 360)
    p.text("<b>People and vehicles (profiler)</b> · starts from step 4 when Frigate reports a person, car, truck or "
           "motorcycle", 46, y3 + 8, 1200, 24, size=13, color="#2E7D45")
    lane = [("Wait for event end", "≤ 10 s person · 20 s vehicle, then pull the clip", "gpu", "clock"),
            ("Detect & track", "per-frame boxes", "gpu", "scan-search"),
            ("Appearance re-ID", "OSNet / DINOv2 fingerprints", "gpu", "user-search"),
            ("Body", "pose · height · gait · actions", "gpu", "footprints"),
            ("Face (enrolled only)", "quality-gated templates", "gpu", "scan-face"),
            ("Attributes", "clothing / vehicle colour & type", "gpu", "eye")]
    lids = []
    for k, (t, sub, role, ic) in enumerate(lane):
        lids.append(p.box(t, sub, 60 + k * 258, y3 + 50, 232, 88, role, ic))
    for a_, b_ in zip(lids, lids[1:]):
        p.edge(a_, b_)
    sight = p.box("Publish sighting", "squirrelguard/<cam>/sighting · /vehicle · /behavior · /fusion", 60 + 5 * 258, y3 + 220, 232, 100, "gpu", "workflow")
    match = p.box("Bridge: match & explain", "re-ID + enrolled face + height + routine → band and reasons", 60 + 3 * 258, y3 + 226, 380, 88, "recorder", "scale")
    pui = p.box("Persons UI / 3D replay", "a person reviews; nothing is acted on automatically", 60 + 1 * 258, y3 + 226, 340, 88, "recorder", "users")
    p.edge(lids[-1], sight)
    p.edge(sight, match)
    p.edge(match, pui)
    p.legend(30, 860, ("camera", "recorder", "gpu", "external"))
    return p


# --------------------------------------------------------------------------------------------- page 3: deployment
def page_deploy():
    p = Page("3 Deployment and trust", 1654, 1140)
    p.header("Deployment and trust boundaries", "Machine roles, network zones and what is allowed to talk to what.")

    def zone(title, x, y, w, h, role):
        fill, stroke = ROLE[role]
        p.add("", f"rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;dashed=1;", x, y, w, h)
        p.text(f"<b>{title}</b>", x + 12, y + 6, w - 24, 22, size=13, color=stroke)

    R = [140, 270, 400, 520, 650]               # row tops inside the zones (boxes are 64 high)
    C = [450, 690, 930]                         # recorder-host columns (boxes 220 wide)
    bw, bh = 220, 64

    zone("Camera network (separate Wi-Fi subnet)", 30, 90, 360, 340, "camera")
    cam = p.box("IP cameras", "Thingino firmware · RTSP · speaker", 50, R[0], 320, bh, "camera", "cctv")
    spk = p.box("deter.sh (forced command)", "plays one of four fixed sounds; validates input", 50, R[1], 320, bh, "camera", "volume-2")
    p.text("Cameras have no route to the analysis machines; only the recorder host holds their credentials.",
           50, 352, 320, 56, size=10, color=GREY)

    zone("Recorder host (CPU only, always on)", 430, 90, 760, 650, "recorder")
    g2 = p.box("go2rtc", "RTSP relay (authenticated)", C[0], R[0], bw, bh, "recorder", "radio")
    rec = p.box("Recorder + janitor", "MKV segments, 7-day retention", C[1], R[0], bw, bh, "recorder", "hard-drive")
    mq = p.box("Mosquitto (MQTT)", "password auth", C[0], R[1], bw, bh, "recorder", "workflow")
    br = p.box("Bridge", "matching, rules, storage", C[1], R[1], bw, bh, "recorder", "git-branch")
    nt = p.box("ntfy", "push · token auth · deny-all", C[2], R[1], bw, bh, "recorder", "bell")
    ui = p.box("Web UIs", "portal · persons · replay · Grafana", C[0], R[2], bw, bh, "recorder", "monitor")
    pg = p.box("PostgreSQL", "loopback + TLS", C[1], R[2], bw, bh, "recorder", "database")
    ng = p.box("nginx + Authelia SSO", "TLS · 2FA · passkeys · default deny", C[0], R[3], bw, bh, "recorder", "key-round")
    p.box("Inline IPS", "Suricata on the host uplink", C[0], R[4], bw, bh, "recorder", "shield")
    mir = p.box("Traffic mirror", "GRE copy of LAN + Wi-Fi", C[1], R[4], bw, bh, "recorder", "network")

    zone("GPU server", 1230, 90, 394, 650, "gpu")
    gx, gw = 1250, 354
    fr = p.box("Frigate NVR", "detection · clips · review UI", gx, R[0], gw, bh, "gpu", "video")
    sg = p.box("Listener + Profiler", "squirrel-guard containers (CPU inference)", gx, R[1], gw, bh, "gpu", "squirrel")
    llm = p.box("Local vision-language model", "llama.cpp · OpenAI-compatible API on the LAN", gx, R[2], gw, bh, "gpu", "brain")
    ids = p.box("Passive IDS", "Suricata on the mirror · severity-1 alerts to ntfy", gx, R[4], gw, bh, "gpu", "shield")

    zone("Operator workstation (optional)", 30, 470, 360, 270, "operator")
    p.box("Scene worker", "3D solve + splats on an NVIDIA GPU; pulls clips over SSH", 50, R[3], 320, bh, "operator", "boxes")
    brw = p.box("Browser / phone", "signs in through SSO; receives push", 50, R[4], 320, bh, "operator", "monitor")

    zone("Outside", 30, 780, 1594, 150, "external")
    p.box("SMTP relay", "optional email alerts from the bridge", 50, 830, 320, bh, "external", "mail")
    p.box("DNS provider", "ACME DNS-01 certificate renewal only", 430, 830, 320, bh, "external", "lock")
    p.text("No video, image, face or identity data leaves the local network. Models run locally; the only outbound "
           "traffic is certificate renewal, optional email and software updates.", 790, 826, 800, 70, size=11)

    p.edge(cam, g2, "RTSP (TCP)")
    p.edge(g2, rec, "RTSP")
    p.edge(g2, fr, "RTSP restream", exit=(0.5, 0), entry=(0, 0.25), points=((560, 118), (1222, 118), (1222, 156)))
    p.edge(fr, mq, "MQTT 1883", exit=(0, 0.75), entry=(0.5, 0), points=((1212, 188), (1212, 228), (560, 228)))
    p.edge(mq, br, "squirrelguard/*")
    p.edge(br, nt, "push")
    p.edge(mq, sg, "frigate/events", exit=(0.25, 1), entry=(0, 0.75), points=((505, 352), (1218, 352), (1218, 318)))
    p.edge(sg, llm, "HTTP API (LAN)")
    p.edge(br, pg, "SQL")
    p.edge(ui, pg, "read")
    p.edge(ng, ui, "auth_request → user header")
    p.edge(br, spk, "SSH · pinned key · one command", color="#B36B00", exit=(0.5, 0), entry=(1, 0.5),
           points=((800, 250), (410, 250), (410, 302)))
    p.edge(brw, ng, "HTTPS", exit=(1, 0.5), entry=(0, 0.5), points=((410, 682), (410, 552)))
    p.edge(ng, fr, "HTTPS + proxy secret", exit=(1, 0.5), entry=(0, 0.8), points=((1200, 552), (1200, 191)))
    p.edge(mir, ids, "GRE mirror")
    p.legend(1380, 950, ("camera", "recorder", "gpu", "operator", "external"))
    return p


# --------------------------------------------------------------------------------------------- page 4: privacy
def page_privacy():
    p = Page("4 Privacy and retention", 1654, 1060)
    p.header("Privacy controls and data retention", "What is kept, for how long, who can see it, and the rules the software enforces.")
    # retention timeline
    p.text("<b>Retention (as configured in the reference deployment)</b>", 30, 90, 900, 24, size=14, color=NAVY)
    rows = [("Push-notification cache", 3, "72 hours", "bell"),
            ("Continuous recordings", 7, "7 days (or a 350 GB cap)", "hard-drive"),
            ("NVR detection clips", 7, "7 days", "video"),
            ("NVR alert clips", 14, "14 days", "film"),
            ("Unknown people: sightings, crops, face data", 90, "90 days", "user-search"),
            ("Occupants of non-household vehicles", 90, "90 days (feature off by default)", "car"),
            ("3D scenes of a visit", 90, "follow the sighting (90 days)", "boxes"),
            ("Enrolled people (with consent)", 100, "until the person is removed", "users")]
    x0, scale = 470, 11.0
    for k, (name, days, label, ic) in enumerate(rows):
        y = 130 + k * 46
        p.add("", f"shape=image;image={icon_uri(ic)};imageAspect=1;", 30, y + 6, 22, 22)
        p.text(name, 62, y, 400, 34, size=12)
        w = days * scale if days < 100 else 990
        fill = "#2E6DA4" if days < 100 else "#6B3FA0"
        p.add("", f"rounded=1;arcSize=30;fillColor={fill};strokeColor=none;opacity={'85' if days < 100 else '35'};", x0, y + 8, w, 18)
        p.text(label, x0 + min(w, 990) + 10 if days < 100 else x0 + 12, y, 360, 34, size=11,
               color=INK if days < 100 else "#3B1F66")
    for d in (0, 7, 14, 30, 60, 90):
        x = x0 + d * scale
        p.text(f"{d} d", x - 20, 496, 40, 18, size=10, color=GREY, align="center")

    # rules
    p.text("<b>Rules the software enforces</b>", 30, 540, 900, 24, size=14, color=NAVY)
    rules = [("scan-face", "Names only for enrolled people", "A person is named only after the operator enrols them with a consent tick-box. Everyone else is \"Unknown #n\"."),
             ("search", "No outside lookups", "No face, plate or identity is ever searched in an outside database or service."),
             ("scale", "No sensitive inference", "No race or ethnicity anywhere. Age band and gender are shown only as labelled apparent estimates, never used for identity."),
             ("shield-check", "Liveness before trust", "A face must pass a liveness check before it can mark someone as household (a photo cannot silence alerts)."),
             ("eye", "Enhanced images stay separate", "Enhanced or super-resolved images are labelled ENHANCED and never fed to face or plate recognition."),
             ("file-text", "Every decision is explained", "Each match stores a band, an estimated percent, reason codes and model versions. Gait, height and attributes can shift a band, never decide it."),
             ("bell", "Alerts only for unknown or watch-listed", "Household and known people never trigger alerts. Actions are always taken by a person."),
             ("key-round", "Single sign-on for every page", "2-factor sign-in for administrators; a restricted group sees only the NVR. Responses are marked private, no-store.")]
    for k, (ic, t, s) in enumerate(rules):
        col, row = k % 2, k // 2
        x, y = 30 + col * 800, 580 + row * 100
        p.add("", "rounded=1;arcSize=6;fillColor=#F7F9FB;strokeColor=#C9D1D9;", x, y, 780, 88)
        p.add("", f"shape=image;image={icon_uri(ic)};imageAspect=1;", x + 16, y + 26, 34, 34)
        p.text(f"<b>{t}</b><br><font color=\"{GREY}\">{s}</font>", x + 66, y + 6, 700, 76, size=12)
    return p


PAGES = [page_stack, page_pipeline, page_deploy, page_privacy]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = [f() for f in PAGES]
    try:
        import analysis_page  # noqa: F401  (optional extra pages, see make_analysis_page.py)
        pages += [pg() for pg in analysis_page.PAGES]
    except ImportError:
        pass
    doc = '<mxfile host="squirrel-guard" type="device">' + "".join(p.xml(f"p{k}") for k, p in enumerate(pages)) + "</mxfile>"
    f = OUT / "squirrel-guard-architecture.drawio"
    f.write_text(doc)
    print("wrote", f, len(pages), "pages")
    if "--export" in sys.argv:
        for k, p in enumerate(pages):
            stem = OUT / p.name.lower().replace(" ", "-")
            for fmt in ("png", "svg"):
                extra = ["-s", "2"] if fmt == "png" else []
                subprocess.run(["drawio", "-x", "-f", fmt, "-p", str(k + 1), *extra, "-o", f"{stem}.{fmt}", str(f),
                                "--no-sandbox"], check=True, capture_output=True, timeout=120)
            print("exported", stem.name)


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    main()
