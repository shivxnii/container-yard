"""Container Yard dashboard - UI polish patcher.
Put this file in the same folder as dashboard.html and double-click it
(or: python apply_ui_polish.py). Logic is untouched, only CSS is injected.
A backup dashboard_before_ui.html is made first."""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "dashboard.html")
BK = os.path.join(HERE, "dashboard_before_ui.html")

JS = r"""
// ===== UI POLISH =====
(function () {
  const s = document.createElement('style'); s.id = 'uiPolish';
  s.textContent = `
:root{--sh:0 4px 16px rgba(10,25,40,.16)}
body{font-family:Inter,"Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
aside{background:linear-gradient(180deg,var(--pn),var(--bg))}
aside a{border-left:3px solid transparent;transition:background .15s,color .15s,border-color .15s}
aside a:hover{background:var(--gl);color:var(--g)}
aside a.on{border-left-color:var(--g)}
.sys{border-radius:10px;background:var(--tile)}
header{box-shadow:0 1px 0 var(--line),0 8px 20px rgba(0,0,0,.10)}
.panel{border-radius:14px;box-shadow:var(--sh);transition:border-color .2s,box-shadow .2s}
.panel:hover{border-color:color-mix(in srgb,var(--g) 45%,var(--line))}
.panel h2{font-size:12px;letter-spacing:.09em;color:var(--mute)}
button,.btn{transition:transform .08s,filter .15s,box-shadow .15s}
button:hover,.btn:hover{filter:brightness(1.1)}
button:active{transform:translateY(1px) scale(.98)}
.ghost:hover{box-shadow:0 0 0 3px color-mix(in srgb,var(--g) 20%,transparent)}
input:focus,select:focus{outline:none;border-color:var(--g)!important;box-shadow:0 0 0 3px color-mix(in srgb,var(--g) 22%,transparent)}
.tiles div{border:1px solid var(--line);transition:transform .15s,border-color .15s}
.tiles div:hover{transform:translateY(-1px);border-color:var(--g)}
.tiles b{letter-spacing:.01em}
.mapwrap{box-shadow:inset 0 12px 40px rgba(0,0,0,.14)}
.hud{background:color-mix(in srgb,var(--pn) 88%,transparent);backdrop-filter:blur(10px);box-shadow:var(--sh);border-radius:12px}
.leg{backdrop-filter:blur(6px);border-radius:10px}
.badge{border:1px solid color-mix(in srgb,var(--g) 35%,transparent)}
.dot.on{box-shadow:0 0 0 4px rgba(31,191,117,.22)}
.bar{height:6px}.bar i{background:linear-gradient(90deg,var(--g),#2bbee8)}
.aimbig b{text-shadow:0 0 18px color-mix(in srgb,var(--g) 40%,transparent)}
.video{box-shadow:0 0 0 1px var(--line)}
.lv{transition:border-color .15s,background .15s}.lv:hover{border-color:var(--g)}
tbody tr:hover td{background:var(--gl)}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:var(--line);border-radius:8px;border:2px solid transparent;background-clip:padding-box}
::-webkit-scrollbar-thumb:hover{background:var(--mute);background-clip:padding-box}
`;
  document.head.appendChild(s);
})();
"""

def main():
    if not os.path.exists(F):
        print("dashboard.html is not in this folder:", HERE); return
    with open(F, "r", encoding="utf-8", newline="") as fh:
        s = fh.read()
    if "UI POLISH" in s:
        print("UI polish already added."); return
    i = s.rfind("</script>")
    if i < 0:
        print("Could not find </script> in dashboard.html - nothing changed."); return
    shutil.copy2(F, BK)
    with open(F, "w", encoding="utf-8", newline="") as fh:
        fh.write(s[:i] + JS + "\n" + s[i:])
    print("UI polish added. Backup:", BK)
    print("Now press Ctrl+F5 in the browser (no server restart needed).")

if __name__ == "__main__":
    main()
    if sys.platform.startswith("win") and sys.stdin and sys.stdin.isatty():
        input("Press Enter to close...")
        