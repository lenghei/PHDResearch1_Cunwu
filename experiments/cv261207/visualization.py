"""Local preview of one measured image with GT and predictions."""
import html

def write_preview(out, image, record, predictions):
    import cv2
    from main.labels import VIS_NAMES

    def draw(rows, color):
        display = image.copy()
        for box in rows:
            x, y, w, h = map(round, box['bbox'])
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display, VIS_NAMES[box['class_id']], (x, max(14, y)), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, color, 1, cv2.LINE_AA)
        return display

    for name, rows, color in [('gt', record['objects'], (230, 140, 20)),
                              ('predictions', [p for p in predictions if p['score'] >= .25], (30, 200, 75))]:
        ok, payload = cv2.imencode('.png', draw(rows, color))
        if not ok:
            raise ValueError('Preview encoding failed.')
        with (out / f'{name}.png').open('xb') as stream:
            stream.write(payload.tobytes())
    title = html.escape(record['stem'])
    (out / 'index.html').write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>CV261207 pair preview</title><style>body{{font:16px system-ui;margin:2rem}}
main{{display:grid;gap:16px;grid-template-columns:repeat(2,minmax(0,1fr))}}img{{width:100%}}
@media(max-width:700px){{main{{grid-template-columns:1fr}}}}</style><h1>{title}</h1>
<p>Original-coordinate boxes. Display confidence is 0.25. The evaluation uses 0.001.</p>
<main><section><h2>Source GT</h2><img src="gt.png" alt="Ground truth"></section>
<section><h2>YOLO predictions</h2><img src="predictions.png" alt="Predictions"></section></main>
<p><a href="metrics.json">COCO metrics</a> · <a href="timing.json">Timing</a> ·
<a href="SHA256SUMS.txt">Checksums</a></p></html>""", encoding='utf-8')
