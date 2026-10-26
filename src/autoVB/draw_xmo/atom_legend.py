"""Shared atom-color entries and a footer for static 3D images."""

from pathlib import Path


def atom_legend_entries(symbols, visible_atoms, colors) -> list[dict[str, str]]:
    """List each visible element once, using exactly the atom mesh colors."""
    return [dict(symbol=symbol.capitalize(), color=colors.get(symbol, "#888888"))
            for symbol in sorted({symbols[index].upper() for index in visible_atoms})]


def append_atom_legend_png(output: Path, entries: list[dict[str, str]], *, transparent: bool) -> None:
    """Append one centered, wrapping atom-color key below the entire PNG grid."""
    # Pillow is a dependency of the optional PyVista renderer.
    from PIL import Image, ImageDraw, ImageFont

    if not entries:
        return
    with Image.open(output) as source:
        image = source.copy()
    font = None
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            font = ImageFont.truetype(name, 18)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    measure = ImageDraw.Draw(image)
    padding, row_height, gap = 16, 30, 22
    title = dict(symbol="Atoms", color=None)
    rows = [[]]
    row_width = 0
    for entry in [title, *entries]:
        width = measure.textbbox((0, 0), entry["symbol"], font=font)[2]
        width += 24 if entry["color"] else 0
        if rows[-1] and row_width + gap + width > image.width - 2 * padding:
            rows.append([])
            row_width = 0
        rows[-1].append((entry, width))
        row_width += width + (gap if len(rows[-1]) > 1 else 0)

    background = (255, 255, 255, 0 if transparent else 255)
    result = Image.new("RGBA", (image.width, image.height + 2 * padding + row_height * len(rows)), background)
    result.paste(image, (0, 0))
    draw = ImageDraw.Draw(result)
    for row_index, row in enumerate(rows):
        width = sum(item_width for _, item_width in row) + gap * (len(row) - 1)
        x = (image.width - width) / 2
        y = image.height + padding + row_index * row_height + row_height / 2
        for entry, item_width in row:
            if entry["color"]:
                draw.ellipse((x, y - 7, x + 14, y + 7), fill=entry["color"], outline="#8A9199")
            draw.text((x + (24 if entry["color"] else 0), y), entry["symbol"],
                      font=font, fill="#333333", anchor="lm")
            x += item_width + gap
    if image.mode == "RGB" and not transparent:
        result = result.convert("RGB")
    result.save(output, format="PNG")
