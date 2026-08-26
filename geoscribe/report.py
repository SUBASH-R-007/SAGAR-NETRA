"""Report generation: contacts.json / .csv / .geojson / .kml and a branded PDF.

Every writer takes the same ``list[Contact]`` and is independently usable;
:func:`write_all` produces the full report bundle for one survey.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from geoscribe.contact import Contact, contacts_json_schema

PIPELINE_VERSION = "0.1.0"

SEVERITY_BANDS = (  # (min score, label, KML/HTML color)
    (75.0, "critical", "ff3b30"),
    (50.0, "high", "ff9500"),
    (25.0, "medium", "ffcc00"),
    (0.0, "low", "34c759"),
)


def severity_band(score: float) -> tuple[str, str]:
    for threshold, label, color in SEVERITY_BANDS:
        if score >= threshold:
            return label, color
    return "low", SEVERITY_BANDS[-1][2]


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def write_contacts_json(contacts: list[Contact], path: str | Path, survey: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "survey": survey,
        "generated_at": _now_iso(),
        "pipeline_version": PIPELINE_VERSION,
        "contacts": [c.model_dump(mode="json") for c in contacts],
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def write_json_schema(path: str | Path) -> Path:
    """Publish the JSON Schema the contacts.json document validates against."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contacts_json_schema(), indent=2), encoding="utf-8")
    return path


def write_contacts_csv(contacts: list[Contact], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "id", "class", "confidence_pct", "severity", "lat", "lon",
        "length_m", "width_m", "height_m", "depth_m", "highlight", "shadow",
        "physics_violation", "side", "ping0", "ping1", "col0", "col1",
        "detected_at", "review", "survey",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for c in contacts:
            writer.writerow([
                c.id, c.cls, c.confidence, c.severity, c.lat, c.lon,
                c.dims.length_m, c.dims.width_m, c.dims.height_m, c.depth_m,
                c.physics.highlight, c.physics.shadow, c.physics.physics_violation,
                c.pixel.side, c.pixel.ping0, c.pixel.ping1, c.pixel.col0, c.pixel.col1,
                c.detected_at, c.review.value, c.survey,
            ])
    return path


def write_contacts_geojson(contacts: list[Contact], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for c in contacts:
        label, color = severity_band(c.severity)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [c.lon, c.lat]},
                "properties": {
                    "id": c.id,
                    "class": c.cls,
                    "confidence_pct": c.confidence,
                    "severity": c.severity,
                    "severity_band": label,
                    "marker-color": f"#{color}",
                    "length_m": c.dims.length_m,
                    "width_m": c.dims.width_m,
                    "height_m": c.dims.height_m,
                    "depth_m": c.depth_m,
                    "physics_violation": c.physics.physics_violation,
                    "review": c.review.value,
                    "footprint": [[lo, la] for la, lo in c.corners],
                },
            }
        )
    doc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def write_contacts_kml(contacts: list[Contact], path: str | Path, survey: str) -> Path:
    import simplekml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kml = simplekml.Kml(name=f"SAGAR-NETRA contacts — {survey}")
    for c in contacts:
        label, color = severity_band(c.severity)
        # KML colors are aabbggrr.
        r, g, b = color[0:2], color[2:4], color[4:6]
        point = kml.newpoint(name=f"{c.id} {c.cls}", coords=[(c.lon, c.lat)])
        point.style.iconstyle.color = f"ff{b}{g}{r}"
        point.style.iconstyle.scale = 1.1
        height = "n/a" if c.dims.height_m is None else f"{c.dims.height_m} m"
        point.description = (
            f"class: {c.cls}\nconfidence: {c.confidence}%\n"
            f"severity: {c.severity} ({label})\n"
            f"size: {c.dims.length_m} x {c.dims.width_m} m, height {height}\n"
            f"depth: {c.depth_m} m\ndetected: {c.detected_at}\n"
            f"physics: highlight={c.physics.highlight} shadow={c.physics.shadow}"
            + (f"\nVIOLATION: {c.physics.violation_reason}" if c.physics.physics_violation else "")
        )
        if c.corners:
            poly = kml.newpolygon(
                name=f"{c.id} footprint",
                outerboundaryis=[(lo, la) for la, lo in [*c.corners, c.corners[0]]],
            )
            poly.style.linestyle.color = f"ff{b}{g}{r}"
            poly.style.linestyle.width = 2
            poly.style.polystyle.color = f"33{b}{g}{r}"
    kml.save(str(path))
    return path


def write_report_pdf(
    contacts: list[Contact], path: str | Path, survey: str
) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
    )
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SNTitle", parent=styles["Title"], textColor=colors.HexColor("#0b3d6b")
    )
    h2 = styles["Heading2"]
    body = styles["BodyText"]

    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm, title=f"SAGAR-NETRA report — {survey}",
    )
    story = [
        Paragraph("SAGAR-NETRA — Marine Debris Contact Report", title_style),
        Paragraph(
            f"Survey: <b>{survey}</b> &nbsp;|&nbsp; generated {_now_iso()} &nbsp;|&nbsp; "
            f"pipeline v{PIPELINE_VERSION} &nbsp;|&nbsp; {len(contacts)} contacts",
            body,
        ),
        Spacer(1, 6 * mm),
    ]

    # Summary table.
    rows = [["ID", "Class", "Conf %", "Severity", "Lat", "Lon", "L×W (m)", "H (m)"]]
    for c in contacts:
        rows.append([
            c.id, c.cls, f"{c.confidence:.0f}", f"{c.severity:.0f}",
            f"{c.lat:.5f}", f"{c.lon:.5f}",
            f"{c.dims.length_m:.1f}x{c.dims.width_m:.1f}",
            "-" if c.dims.height_m is None else f"{c.dims.height_m:.1f}",
        ])
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d6b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ])
    )
    story += [table, Spacer(1, 8 * mm)]

    # Per-contact detail pages.
    for c in contacts:
        band, _ = severity_band(c.severity)
        story.append(Paragraph(f"{c.id} — {c.cls} ({c.confidence:.0f}%)", h2))
        height = "n/a" if c.dims.height_m is None else f"{c.dims.height_m} m"
        facts = (
            f"Position: {c.lat:.6f}, {c.lon:.6f} &nbsp;|&nbsp; depth {c.depth_m} m<br/>"
            f"Dimensions: {c.dims.length_m} × {c.dims.width_m} m, height {height}<br/>"
            f"Severity: {c.severity} ({band}) &nbsp;|&nbsp; detected {c.detected_at}<br/>"
            f"Acoustic cues: highlight={c.physics.highlight}, shadow={c.physics.shadow}"
            f" (len {c.physics.shadow_len_m} m)"
            + (
                f"<br/><b>Physics violation:</b> {c.physics.violation_reason}"
                if c.physics.physics_violation
                else ""
            )
        )
        story.append(Paragraph(facts, body))
        image_path = c.evidence_png or c.thumbnail_png
        if image_path and Path(image_path).exists():
            from PIL import Image as PILImage

            with PILImage.open(image_path) as im:
                w, h = im.size
            max_w = 150 * mm
            scale = min(max_w / w, (90 * mm) / h)
            story.append(RLImage(image_path, width=w * scale, height=h * scale))
        story.append(Spacer(1, 6 * mm))

    doc.build(story)
    return path


def write_all(
    contacts: list[Contact], out_dir: str | Path, survey: str
) -> dict[str, Path]:
    """The full report bundle; returns format -> written path."""
    out_dir = Path(out_dir)
    return {
        "json": write_contacts_json(contacts, out_dir / "contacts.json", survey),
        "schema": write_json_schema(out_dir / "contacts.schema.json"),
        "csv": write_contacts_csv(contacts, out_dir / "contacts.csv"),
        "geojson": write_contacts_geojson(contacts, out_dir / "contacts.geojson"),
        "kml": write_contacts_kml(contacts, out_dir / "contacts.kml", survey),
        "pdf": write_report_pdf(contacts, out_dir / "report.pdf", survey),
    }
