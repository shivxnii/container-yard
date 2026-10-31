// ===== ENTRY ON RIGHT WALL (bottom) =====
(function () {
  const GY0 = .4, GW = 1.6, GH = 1.5, cy = GY0 + GW / 2;
  try { const s = document.querySelector('.sys'); if (s) s.innerHTML = s.innerHTML.replace('v1.7.0', 'v1.8 entry-right'); } catch (e) { }
  const oB = buildRoom;
  buildRoom = function () {
    oB();
    if (!layout || !gRoom) return;
    try {
      const tex = texC['ENTRY#0f8a5f#fff26'] && texC['ENTRY#0f8a5f#fff26'].t;
      gRoom.children.slice().forEach(o => {
        if (o.isSprite && tex && o.material.map === tex) gRoom.remove(o);
        else if (o.isMesh && o.geometry.type === 'PlaneGeometry' && o.material.color && o.material.color.getHex() === 0x9fd3e6) gRoom.remove(o);
        else if (o.isMesh && o.geometry.type === 'BoxGeometry' && Math.abs(o.position.x - (W + .1)) < .01 && Math.abs(o.geometry.parameters.depth - (H + .4)) < .01) { gRoom.remove(o); walls = walls.filter(w => w.m !== o.material); }
      });
      const mk = (w, h, d, x, y, z, col, test) => { const m = new T.MeshLambertMaterial({ color: col, transparent: true }), b = new T.Mesh(new T.BoxGeometry(w, h, d), m); b.position.set(x, y, z); gRoom.add(b); if (test) walls.push({ m, test }); };
      const rt = p => p.x > W, y1 = GY0 + GW;
      mk(doorW, .95, .2, doorX(), .475, .1, 0xdfe4e9);
      mk(.2, wh, GY0 + .2, W + .1, wh / 2, -(GY0 - .2) / 2, 0xe5eaef, rt);
      mk(.2, wh, H + .2 - y1, W + .1, wh / 2, -(y1 + H + .2) / 2, 0xe5eaef, rt);
      mk(.2, wh - GH, GW, W + .1, GH + (wh - GH) / 2, -cy, 0xe5eaef, rt);
      const fl = new T.Mesh(new T.PlaneGeometry(1, GW), new T.MeshBasicMaterial({ color: 0x0f8a5f, transparent: true, opacity: .35 }));
      fl.rotation.x = -Math.PI / 2; fl.position.set(W - .5, .02, -cy); gRoom.add(fl);
      const en = sprite('ENTRY', '#0f8a5f', '#fff', 26); en.position.set(W + .1, wh + .35, -cy); gRoom.add(en);
    } catch (e) { console.error('entry', e); }
  };
  const oS = setTarget;
  setTarget = function () {
    const wasIdle = lastStatus === 'idle';
    oS();
    if (wasIdle && dev && layout) {
      dev.x = W - .5; dev.y = cy; trail = [[dev.x, dev.y]];
      if (trailGeo) { const a = trailGeo.attributes.position.array; a[0] = dev.x; a[1] = .04; a[2] = -dev.y; trailGeo.attributes.position.needsUpdate = true; trailGeo.setDrawRange(0, 1); }
    }
  };
  const oR = renderDims;
  renderDims = function () {
    oR();
    try { const el = document.getElementById('dimHud'); if (el && layout && /Entry door: (?!right wall)/.test(el.innerHTML)) el.innerHTML = el.innerHTML.replace(/Entry door:[^<]*/, 'Entry door: right wall, bottom · Y 0.4–2.0 m'); } catch (e) { }
  };
  lastDims = '';
})();