# Usage: put this file next to dashboard.html, then run:  python apply_patch.py
# It reads dashboard.html and writes a NEW complete file, dashboard_new.html (the original is not touched).
# Option: python apply_patch.py --no-entry   = only add the always-on pointers + keep-containers patches
# Patches already present in dashboard.html are skipped, so it is safe to run again.
import sys
F = 'dashboard.html'

P1_MARK = '// ===== always-on pointers'
P1 = r'''
// ===== always-on pointers: camera, pointer and pin stay on the map =====
(function () {
  let lastAim = null;
  function keep() {
    const e = 2 * orb.r * Math.tan(Math.PI / 9) / Math.max(200, R.domElement.clientHeight);
    if (aimS && aimInfo) lastAim = { x: aimS.x, y: aimS.y, h: aimS.h, hm: aimInfo.h, lv: aimInfo.level };
    if (!aimInfo && !lastAimInfo) lastAim = null;
    if (!gAimM.g.visible && lastAim && cfg.dot)
      placeMark(gAimM, lastAim.x, lastAim.y, zS(lastAim.h), 'LAST POINTER  ' + lastAim.hm.toFixed(2) + ' m · L' + lastAim.lv, e, 26);
    if (!gPinM.g.visible && pinT && pinInfo)
      placeMark(gPinM, pinT.x, pinT.y, zS(pinT.h), 'PIN  ' + pinInfo.h.toFixed(2) + ' m · L' + pinInfo.level, e, 56);
    if (!dev && lastDev && cfg.dot) {
      const dz = Math.max(zS(lastDev.z), .05), pu = .5 + .5 * Math.sin(performance.now() / 300);
      gDot.visible = true; gDot.position.set(lastDev.x, 0, -lastDev.y);
      sp.scale.setScalar(9 * e); sp.position.y = 9 * e;
      halo.scale.setScalar((26 + 8 * pu) * e); ring.scale.setScalar((16 + 6 * pu) * e);
      pole.scale.set(2.2 * e, dz, 2.2 * e); pole.position.y = dz / 2;
      capS.scale.setScalar(5 * e); capS.position.y = dz;
      devLbl.scale.set(26 * e * 520 / 56, 26 * e, 1); devLbl.position.y = dz + 28 * e;
      const tx = `LAST CAMERA  X ${lastDev.x.toFixed(1)}  Y ${lastDev.y.toFixed(1)}  Z ${lastDev.z.toFixed(2)} m`;
      if (tx !== lastDevTxt) { lastDevTxt = tx; paintDev(tx); }
    }
  }
  const t = setInterval(() => {
    if (typeof R === 'undefined' || !R || !R.render) return;
    if (R.__keep) { clearInterval(t); return; }
    const orig = R.render.bind(R);
    R.render = (s, c) => { try { keep(); } catch (err) { } return orig(s, c); };
    R.__keep = 1; clearInterval(t);
  }, 300);
})();
'''

P2_MARK = '// ===== keep captured containers'
P2 = r'''
// ===== keep captured containers: once the video has captured a container it stays on the map =====
(function () {
  const seen = new Map();                       // container_id -> last known record
  const orig = makeView;
  makeView = function () {
    try {
      const now = new Set();
      (raw || []).forEach(c => { now.add(c.container_id); seen.set(c.container_id, Object.assign({}, c)); });
      seen.forEach((c, id) => { if (!now.has(id)) c.in_view = false; });   // not in the camera now -> shown faded, but still there
      raw = Array.from(seen.values());
    } catch (e) { }
    return orig();
  };
  const b = document.getElementById('lvClear'); if (b) b.addEventListener('click', () => seen.clear());   // "Clear map" empties it
})();
'''


P3_MARK = '// ===== entry left v2'
P3 = r'''
// ===== entry left v2 + red dot in sync with the video =====
(function () {
  try { const sy = document.querySelector('.sys'); if (sy) sy.innerHTML = sy.innerHTML.replace(/v1\.[0-9.]+/, 'v1.8 patched'); } catch (e) { }
  // one-time move of the entry door to the LEFT end of the front (bottom) wall, even if "right" was saved earlier
  try {
    if (!localStorage.getItem('yardEntryLeft2')) { cfg.entry = 'left'; saveCfg(); localStorage.setItem('yardEntryLeft2', '1'); if (typeof R !== 'undefined' && R && layout) { buildRoom(); lastDims = ''; } }
  } catch (e) { }
  // when the video starts, the camera dot and the red path start AT the door immediately (no walk-in delay)
  const oSet = setTarget;
  setTarget = function () {
    const wasIdle = lastStatus === 'idle';
    oSet();
    if (wasIdle && dev && layout) {
      dev.x = doorX(); dev.y = .3;
      if (trailGeo) { trail = [[dev.x, dev.y]]; const a = trailGeo.attributes.position.array; a[0] = dev.x; a[1] = .04; a[2] = -dev.y; trailGeo.attributes.position.needsUpdate = true; trailGeo.setDrawRange(0, 1); }
    }
  };
  // the dot follows the video position quickly (about 0.1 s behind) instead of crawling at walking speed
  moveDot = function (dt) {
    if (!dev || !tgt) return;
    let gx = tgt.x, gy = tgt.y; if (layout && dev.y < .3 && gy > .3) { gx = doorX(); gy = .6; }
    const dx = gx - dev.x, dy = gy - dev.y, dist = Math.hypot(dx, dy);
    if (dist > .02) {
      const step = Math.min(dist, Math.max(dist * Math.min(1, dt * 12), 8 * dt)); dev.x += dx / dist * step; dev.y += dy / dist * step;
      if (!hasHeading && step / dt > .15) dir = [dir[0] * .85 + dx / dist * .15, dir[1] * .85 + dy / dist * .15];
    }
    dev.z += (tgt.z - dev.z) * Math.min(1, dt * 8);
    const l = trail[trail.length - 1]; if (!l || Math.hypot(dev.x - l[0], dev.y - l[1]) > .08) {
      trail.push([dev.x, dev.y]); if (trail.length > TRAIL_MAX) trail.shift();
      const a = trailGeo.attributes.position.array; trail.forEach((p, i) => { a[i * 3] = p[0]; a[i * 3 + 1] = .04; a[i * 3 + 2] = -p[1]; }); trailGeo.attributes.position.needsUpdate = true; trailGeo.setDrawRange(0, trail.length);
    }
  };
})();
'''

P4_MARK = '// ===== entry on right wall'
P4 = r'''
// ===== entry on right wall: gap at the bottom of the RIGHT side wall, front-wall gap filled =====
(function () {
  const GY0 = .4, GW = 1.6, GH = 1.5;                 // gap on the right wall: Y 0.4-2.0 m, 1.5 m high
  const entryPt = () => ({ x: W - .5, y: GY0 + GW / 2 });
  const oB = buildRoom;
  buildRoom = function () {
    oB();
    if (!layout || !gRoom) return;
    try {
      const entTex = texC['ENTRY#0f8a5f#fff26'] && texC['ENTRY#0f8a5f#fff26'].t;
      gRoom.children.slice().forEach(o => {
        if (o.isSprite && entTex && o.material.map === entTex) gRoom.remove(o);                                   // old ENTRY label
        else if (o.isMesh && o.geometry.type === 'PlaneGeometry' && o.material.color && o.material.color.getHex() === 0x9fd3e6) gRoom.remove(o);   // old glass
        else if (o.isMesh && o.geometry.type === 'BoxGeometry' && Math.abs(o.position.x - (W + .1)) < .01 && Math.abs(o.geometry.parameters.depth - (H + .4)) < .01) {
          gRoom.remove(o); walls = walls.filter(w => w.m !== o.material);                                          // old solid right wall
        }
      });
      const mk = (w, h, d, x, y, z, col, test) => {
        const m = new T.MeshLambertMaterial({ color: col, transparent: true }), b = new T.Mesh(new T.BoxGeometry(w, h, d), m);
        b.position.set(x, y, z); gRoom.add(b); if (test) walls.push({ m, test }); return b;
      };
      const R_ = p => p.x > W, y1 = GY0 + GW;
      mk(doorW, .95, .2, doorX(), .475, .1, 0xdfe4e9);                                           // front (bottom) wall: old entry gap filled
      mk(.2, wh, GY0 + .2, W + .1, wh / 2, -(GY0 - .2) / 2, 0xe5eaef, R_);                        // right wall below the gap
      mk(.2, wh, H + .2 - y1, W + .1, wh / 2, -(y1 + H + .2) / 2, 0xe5eaef, R_);                  // right wall above the gap
      mk(.2, wh - GH, GW, W + .1, GH + (wh - GH) / 2, -(GY0 + GW / 2), 0xe5eaef, R_);             // lintel over the gap
      const fl = new T.Mesh(new T.PlaneGeometry(1, GW), new T.MeshBasicMaterial({ color: 0x0f8a5f, transparent: true, opacity: .35 }));
      fl.rotation.x = -Math.PI / 2; fl.position.set(W - .5, .02, -(GY0 + GW / 2)); gRoom.add(fl);   // green floor mat at the entry
      const en = sprite('ENTRY', '#0f8a5f', '#fff', 26); en.position.set(W + .1, wh + .35, -(GY0 + GW / 2)); gRoom.add(en);
    } catch (e) { console.error('entry patch', e); }
  };
  // camera dot + red path start at the right-wall entry
  const oSet = setTarget;
  setTarget = function () {
    const wasIdle = lastStatus === 'idle';
    oSet();
    if (wasIdle && dev && layout) {
      const p = entryPt(); dev.x = p.x; dev.y = p.y;
      if (trailGeo) { trail = [[dev.x, dev.y]]; const a = trailGeo.attributes.position.array; a[0] = dev.x; a[1] = .04; a[2] = -dev.y; trailGeo.attributes.position.needsUpdate = true; trailGeo.setDrawRange(0, 1); }
    }
  };
  const oR = renderDims;
  renderDims = function () {
    oR();
    try {
      const el = document.getElementById('dimHud');
      if (el && layout && /Entry door: (?!right wall)/.test(el.innerHTML)) el.innerHTML = el.innerHTML.replace(/Entry door:[^<]*/, 'Entry door: right wall, bottom · Y ' + GY0.toFixed(1) + '–' + (GY0 + GW).toFixed(1) + ' m');
    } catch (e) { }
  };
  if (typeof R !== 'undefined' && R && layout) buildRoom();
})();
'''

import os
found = []
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d.lower() not in ('.venv', 'venv', 'node_modules', '.git', '__pycache__')]
    if 'dashboard.html' in files:
        found.append(os.path.join(root, 'dashboard.html'))
if not found:
    sys.exit('No dashboard.html found under this folder')
made = 0
for f in found:
    s = open(f, encoding='utf-8').read()
    todo = [p for m, p in ((P1_MARK, P1), (P2_MARK, P2), (P3_MARK, P3), (P4_MARK, P4)) if not ('--no-entry' in sys.argv and m in (P3_MARK, P4_MARK)) and m not in s]
    if "entry: 'right', dims" in s:
        s = s.replace("entry: 'right', dims", "entry: 'left', dims"); todo = todo or ['']
    i = s.rfind('</script>')
    if not todo or i < 0:
        print('skip (already patched or no script):', f); continue
    out = os.path.join(os.path.dirname(f), 'dashboard_new.html')
    open(out, 'w', encoding='utf-8').write(s[:i] + ''.join(todo) + s[i:])
    print('created', out, '(%d patch(es))' % len(todo)); made += 1
print('Done: %d new file(s). Rename each dashboard_new.html to dashboard.html.' % made)