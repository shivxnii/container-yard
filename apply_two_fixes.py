"""Container Yard dashboard - two small fixes (nothing else is changed):
 1. the blue focus cone / focus window is much narrower
 2. every container that has been seen by the camera once stays on the map
    (until you press "Clear map")
Put this file next to dashboard.html and run:  python apply_two_fixes.py
A backup dashboard_before_fix.html is made first."""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "dashboard.html")
BK = os.path.join(HERE, "dashboard_before_fix.html")

JS = r"""
// ===== FOCUS+KEEP FIX =====
(function () {
  // 1) narrower focus cone: change CONE_K (0.45 = 45 % of the old width; smaller = narrower)
  const CONE_K = 0.45, CONE_MIN = 0.6;
  const oU = updateAim;
  updateAim = function () {
    oU();
    if (aimT) aimT.hw = Math.max(CONE_MIN, aimT.hw * CONE_K);
  };
  // 2) containers stay on the map once the camera has seen them (they are drawn faded when out of view)
  const kept = new Map(), oM = makeView0;
  makeView0 = function () {
    const cur = new Set();
    (raw || []).forEach(c => { if (c && c.container_id && !c._kept) { cur.add(c.container_id); kept.set(c.container_id, c); } });
    const extra = [];
    kept.forEach((c, id) => { if (!cur.has(id)) extra.push({ ...c, _kept: true, live: true, in_view: false, bx: undefined, by: undefined }); });
    if (extra.length) raw = (raw || []).concat(extra);
    return oM();
  };
  const cl = document.getElementById('lvClear'); if (cl) cl.addEventListener('click', () => kept.clear());
})();
"""

def main():
    if not os.path.exists(F):
        print("dashboard.html is not in this folder:", HERE); return
    with open(F, "r", encoding="utf-8", newline="") as fh:
        s = fh.read()
    if "FOCUS+KEEP FIX" in s:
        print("Already added."); return
    i = s.rfind("</script>")
    if i < 0:
        print("Could not find </script> - nothing changed."); return
    shutil.copy2(F, BK)
    with open(F, "w", encoding="utf-8", newline="") as fh:
        fh.write(s[:i] + JS + "\n" + s[i:])
    print("Done. Backup:", BK)
    print("Now press Ctrl+F5 in the browser (no server restart needed).")

if __name__ == "__main__":
    main()
    if sys.platform.startswith("win") and sys.stdin and sys.stdin.isatty():
        input("Press Enter to close...")