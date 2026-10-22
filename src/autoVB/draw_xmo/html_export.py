"""Preserve VB annotations on top of PyVista's offline VTK.js scene export."""

import json
from pathlib import Path


def export_interactive_html(plotter, output: Path, panels: list[dict]) -> None:
    """Export mesh data plus browser-side labels unsupported by VTK.js serialization."""
    document = plotter.trame.export_html(None).getvalue()
    # Script data must not allow an XMO legend to terminate the script element.
    payload = json.dumps(panels, ensure_ascii=False).replace("<", "\\u003c")
    overlay = r"""
<style>
#autovb-tabs {position:fixed;top:0;left:0;right:0;height:52px;box-sizing:border-box;display:flex;gap:6px;align-items:center;padding:6px 8px;overflow-x:auto;background:white;border-bottom:1px solid #ddd;z-index:12;font-family:Arial,sans-serif}
#autovb-tabs button {flex:none;border:1px solid #ccc;border-radius:6px;background:#f5f5f5;color:#333;padding:7px 16px;font-size:14px;cursor:pointer}
#autovb-tabs button[aria-selected="true"] {background:#244f83;border-color:#244f83;color:white}
#autovb-tabs button:focus-visible {outline:2px solid #bb1938;outline-offset:2px}
#autovb-tabs button:disabled {cursor:wait;opacity:0.6}
#autovb-overlay {position:fixed;inset:0;pointer-events:none;z-index:10;font-family:Arial,sans-serif;color:#222}
.autovb-panel {position:absolute;overflow:hidden}
.autovb-legend {position:absolute;bottom:12px;left:12px;white-space:pre-line;font-size:16px}
.autovb-label {position:absolute;transform:translate(-50%,-50%);white-space:nowrap;font-size:14px;text-shadow:0 0 3px white,0 0 3px white}
.autovb-charge {color:#bb1938;font-weight:bold;font-size:21px}
#autovb-help {position:fixed;bottom:0;left:0;right:0;height:32px;box-sizing:border-box;padding:3px 8px;z-index:11;background:#ffffffdf;font:12px Arial,sans-serif;display:flex;gap:16px;align-items:center}
#autovb-help button {cursor:pointer;flex:none}
#autovb-status {white-space:nowrap}
</style>
<div id="autovb-tabs" role="tablist" aria-label="VB structures"></div>
<div id="autovb-overlay"></div>
<div id="autovb-help"><span>Drag: rotate · Wheel: zoom · Shift+drag: pan</span><button id="autovb-reset" type="button">Reset view</button><span id="autovb-status">Loading 3D scene…</span></div>
<script id="autovb-panels" type="application/json">__PANELS__</script>
<script>
(() => {
  const panels = JSON.parse(document.getElementById('autovb-panels').textContent);
  const overlay = document.getElementById('autovb-overlay');
  const tabs = document.getElementById('autovb-tabs');
  const status = document.getElementById('autovb-status');
  let ready = false;
  let activeIndex = 0;
  const entries = panels.map((panel, index) => {
    const element = document.createElement('div');
    element.className = 'autovb-panel'; element.hidden = true;
    element.id = `autovb-panel-${index}`; element.setAttribute('role', 'tabpanel');
    element.setAttribute('aria-labelledby', `autovb-tab-${index}`);
    const legend = document.createElement('div');
    legend.className = 'autovb-legend'; legend.textContent = panel.legend;
    element.append(legend); overlay.append(element);
    const tab = document.createElement('button');
    tab.id = `autovb-tab-${index}`; tab.type = 'button'; tab.textContent = `Rank ${panel.rank}`;
    tab.setAttribute('role', 'tab'); tab.setAttribute('aria-controls', element.id);
    tab.setAttribute('aria-selected', String(index === 0)); tab.tabIndex = index === 0 ? 0 : -1;
    tab.disabled = true; tab.addEventListener('click', () => activate(index));
    tab.addEventListener('keydown', event => {
      const targets = {ArrowRight:(index + 1) % panels.length,
                       ArrowLeft:(index - 1 + panels.length) % panels.length,
                       Home:0, End:panels.length - 1};
      if (Object.hasOwn(targets, event.key)) {
        event.preventDefault(); activate(targets[event.key]); entries[targets[event.key]].tab.focus();
      }
    });
    tabs.append(tab);
    const labels = panel.labels.map(label => {
      const text = document.createElement('div');
      text.className = 'autovb-label' + (label.kind === 'charge' ? ' autovb-charge' : '');
      text.textContent = label.text; element.append(text);
      return {...label, element: text};
    });
    return {...panel, element, tab, labels, renderer: null, initial: null};
  });
  function activate(index) {
    if (!ready) return;
    activeIndex = index;
    const renderWindow = window.global.renderWindow;
    // Detach inactive renderers so that they cannot draw or receive mouse input.
    [...renderWindow.getRenderers()].forEach(renderer => renderWindow.removeRenderer(renderer));
    const entry = entries[index];
    entry.renderer.setViewport(0, 0, 1, 1); renderWindow.addRenderer(entry.renderer);
    renderWindow.getInteractor()?.getInteractorStyle()?.setCurrentRenderer?.(entry.renderer);
    entries.forEach((item, i) => {
      item.element.hidden = i !== index; item.tab.setAttribute('aria-selected', String(i === index));
      item.tab.tabIndex = i === index ? 0 : -1;
    });
    status.textContent = `Rank ${entry.rank} of ${entries.length}`;
    document.documentElement.dataset.autovbActiveRank = String(entry.rank);
    renderWindow.render();
  }
  document.getElementById('autovb-reset').addEventListener('click', () => {
    const entry = entries[activeIndex];
    if (entry?.renderer && entry.initial) {
      entry.renderer.getActiveCamera().set(entry.initial); window.global.renderWindow.render();
    }
  });
  function update() {
    const renderWindow = window.global?.renderWindow;
    const view = renderWindow?.getViews?.()[0];
    const canvas = view?.getCanvas?.();
    if (view && canvas) {
      if (!ready) {
        const renderers = renderWindow.getRenderers();
        entries.forEach(entry => {
          if (!entry.renderer) entry.renderer = renderers.find(renderer => {
            const actual = renderer.getViewport();
            return renderer.getActors().length && entry.viewport.every((value, i) => Math.abs(value - actual[i]) < 1e-6);
          });
          if (!entry.renderer || entry.initial) return;
          const camera = entry.renderer.getActiveCamera();
          // Preserve parameters omitted by native VTK.js scene serialization.
          if (entry.camera) camera.set(entry.camera);
          entry.initial = {
            position:[...camera.getPosition()], focalPoint:[...camera.getFocalPoint()],
            viewUp:[...camera.getViewUp()], parallelScale:camera.getParallelScale(),
            parallelProjection:camera.getParallelProjection(), clippingRange:[...camera.getClippingRange()],
          };
        });
        if (entries.every(entry => entry.initial)) {
          ready = true; entries.forEach(entry => entry.tab.disabled = false); activate(0);
          document.documentElement.dataset.autovbReady = 'true';
        }
      }
      if (ready) {
        const entry = entries[activeIndex];
        const box = canvas.getBoundingClientRect();
        const [width, height] = view.getSize();
        Object.assign(entry.element.style, {left:box.left+'px', top:box.top+'px', width:box.width+'px', height:box.height+'px'});
        entry.labels.forEach(label => {
          const [x, y, z] = view.worldToDisplay(...label.position, entry.renderer);
          let offset = [0, 0];
          if (label.kind === 'charge') {
            // Keep charges outside the atom sphere at any tab size or zoom level.
            const up = entry.renderer.getActiveCamera().getViewUp();
            const edge = label.position.map((value, i) => value + up[i] * (label.radius ?? 0.26));
            const edgeY = view.worldToDisplay(...edge, entry.renderer)[1];
            const gap = Math.abs(edgeY - y) * box.height / height + 12;
            offset = [0.65 * gap, -0.8 * gap];
          }
          const px = x * box.width / width + offset[0];
          const py = box.height - y * box.height / height + offset[1];
          label.element.style.left = px + 'px'; label.element.style.top = py + 'px';
          label.element.style.display = Number.isFinite(px + py + z) && z >= 0 && z <= 1 ? '' : 'none';
        });
      }
    }
    requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
})();
</script>
""".replace("__PANELS__", payload)
    document = document.replace("<title>VTK.js | Example - OfflineLocalView</title>",
                                "<title>autoVB · Interactive VB structures</title>")
    # The offline viewer clears its root container during startup. Keep the scene
    # inside its own element so that loading it cannot remove our annotations.
    document = document.replace("<body>",
                                '<body><div class="content" id="autovb-scene" '
                                'style="position:fixed;inset:52px 0 32px"></div>', 1)
    document = document.replace("</body>", overlay + "\n</body>")
    output.write_text(document, encoding="utf-8")
