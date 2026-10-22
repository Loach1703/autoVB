import io
import json
import re
from unittest.mock import Mock

import numpy as np
import pytest

from autoVB.draw_xmo.atom_legend import append_atom_legend_png, atom_legend_entries
from autoVB.draw_xmo.html_export import export_interactive_html
from autoVB.draw_xmo.pyvista_drawer import ATOM_COLORS


def test_legend_lists_only_visible_elements_once_and_keeps_unknown_element_color():
    symbols = ["C", "H", "C", "N", "Cl", "Xe"]
    entries = atom_legend_entries(symbols, [0, 2, 3, 4, 5], ATOM_COLORS)
    assert entries == [dict(symbol="C", color="#394352"), dict(symbol="Cl", color="#5AB45A"),
                       dict(symbol="N", color="#376FCE"), dict(symbol="Xe", color="#888888")]


@pytest.mark.parametrize("transparent", [False, True])
def test_png_footer_preserves_image_and_draws_atom_colors(tmp_path, transparent):
    Image = pytest.importorskip("PIL.Image")
    output = tmp_path / "figure.png"
    mode = "RGBA" if transparent else "RGB"
    original = Image.new(mode, (400, 180), (240, 240, 240, 0) if transparent else "white")
    original.save(output)
    entries = atom_legend_entries(["C", "H", "N"], [0, 1, 2], ATOM_COLORS)
    append_atom_legend_png(output, entries, transparent=transparent)
    with Image.open(output) as result:
        assert result.width == original.width and result.height > original.height
        assert result.mode == mode
        assert result.crop((0, 0, 400, 180)).tobytes() == original.tobytes()
        footer = np.asarray(result.convert("RGB"))[180:]
        for color in ("#394352", "#E8E9ED", "#376FCE"):
            expected = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
            assert np.any(np.all(footer == expected, axis=-1))
        if transparent:
            assert result.getchannel("A").getextrema() == (0, 255)


def test_html_has_one_shared_atom_key_and_escapes_its_payload(tmp_path):
    plotter = Mock()
    plotter.trame.export_html.return_value = io.StringIO("<html><body>VTK scene</body></html>")
    output = tmp_path / "figure.html"
    entries = [dict(symbol="C", color="#394352"), dict(symbol="</script>", color="#888888")]
    export_interactive_html(plotter, output, [], atom_legend=entries)
    document = output.read_text()
    assert document.count('id="autovb-atom-legend"') == 1
    payload = re.search(r'<script id="autovb-atom-colors" type="application/json">(.*?)</script>', document).group(1)
    assert json.loads(payload) == entries
    assert "</script>" not in payload
