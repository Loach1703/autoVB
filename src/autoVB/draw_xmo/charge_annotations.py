"""Screen-facing charge text and optional rings for static PyVista images."""

import numpy as np

CHARGE_STROKE_RATIO = 0.1


def add_png_charge_labels(plotter, positions, notes, *, color, circle, parallel_scale,
                          panel_size, screen_right, screen_up):
    """Center bold text at each world position; size circles from font metrics."""
    import pyvista as pv
    from vtkmodules.vtkRenderingCore import vtkTextActor, vtkTextRenderer

    rgb = tuple(int(color[index:index + 2], 16) / 255 for index in (1, 3, 5))
    dpi = plotter.render_window.GetDPI()
    for position, note in zip(positions, notes):
        label = note.replace("-", "−")
        actor = vtkTextActor()
        actor.SetInput(label)
        coordinate = actor.GetPositionCoordinate()
        coordinate.SetCoordinateSystemToWorld()
        coordinate.SetValue(*position)
        prop = actor.GetTextProperty()
        prop.SetFontFamilyToArial()
        prop.SetFontSize(30)
        prop.BoldOn()
        prop.SetColor(*rgb)
        prop.SetJustificationToCentered()
        prop.SetVerticalJustificationToCentered()
        plotter.add_actor(actor, reset_camera=False, pickable=False, render=False)
        if circle:
            bounds = [0, 0, 0, 0]
            vtkTextRenderer.GetInstance().GetBoundingBox(prop, label, bounds, dpi)
            radius_px = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) / 2 + 6
            radius = radius_px * 2 * parallel_scale / panel_size
            angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
            points = (position + radius * (np.cos(angles)[:, None] * screen_right
                                           + np.sin(angles)[:, None] * screen_up))
            stroke_px = CHARGE_STROKE_RATIO * prop.GetFontSize() * dpi / 72
            # A tube keeps its screen width when SSAA downsamples the screenshot.
            ring = pv.lines_from_points(points, close=True).tube(
                radius=stroke_px * parallel_scale / panel_size, n_sides=12,
            )
            plotter.add_mesh(ring, color=color, lighting=False, pickable=False, render=False)
