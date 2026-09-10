# @becometen/face-metrics

Pure, dependency-free facial-geometry metrics over MediaPipe's 478-point face mesh.
Face shape, symmetry, golden-ratio deviation, canthal tilt, "hunter eyes", jawline,
facial thirds and fifths, eye shape and pupillary distance, all computed from
landmarks you already have. No model, no network, no image upload: it is a set of
total functions over an array of points.

This is the exact code behind the free on-device tools at
[becometen.com/tools](https://becometen.com/tools). Extracted and MIT-licensed so
anyone can audit how the numbers are produced or reuse them.

## Install

```sh
npm install @becometen/face-metrics
```

## Usage

Get landmarks from `@mediapipe/tasks-vision` (or any source that yields the
canonical 468/478-point mesh, normalized 0–1 in image space), then measure:

```ts
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { measure } from "@becometen/face-metrics";

const vision = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm",
);
const landmarker = await FaceLandmarker.createFromOptions(vision, {
  baseOptions: { modelAssetPath: "face_landmarker.task" },
  runningMode: "IMAGE",
  numFaces: 1,
});

const img = document.querySelector("img")!;
const { faceLandmarks } = landmarker.detect(img);
const points = faceLandmarks[0]; // Array<{x, y, z}>
const aspect = img.naturalWidth / img.naturalHeight;

const result = measure("face-shape", points, aspect);
if (result.gate.ok) {
  console.log(result.value.shape, result.value.match); // "oval", 0.62
} else {
  console.log("refused:", result.gate.reason); // "yaw" | "pitch" | "roll" | "small" | "points"
}
```

`measure(metric, points, aspect, size?)` runs the pose gate first and only then
computes; it never throws. Every result is `{ gate, value, overlay }`, where
`overlay` lists the landmark indices the metric used, so you can draw them.

Pupillary distance also needs the natural pixel size of the image:

```ts
measure("pupillary-distance", points, aspect, { w: img.naturalWidth, h: img.naturalHeight });
```

## Metrics

| Metric key | Function | Returns |
| --- | --- | --- |
| `face-shape` | `faceShape(frame)` | `shape` (oval, round, square, oblong, heart, diamond), `match` 0–1, `runnerUp`, the four ratios, gonial angle |
| `face-symmetry` | `symmetry(frame)` | `score` 0–100 plus per-part scores for eyes, brows, nose, mouth, jaw |
| `golden-ratio` | `goldenRatio(frame)` | `score` 0–100 and each measured ratio with its target |
| `canthal-tilt` | `canthalTiltMetric(frame)` | left, right and mean tilt in degrees (positive = outer canthus higher) |
| `hunter-eyes` | `hunterEyes(frame)` | `score`, eye aspect ratio, brow gap, tilt |
| `jawline-score` | `jawline(frame)` | `score`, left/right/mean gonial angle, chin taper, jaw-to-cheek ratio |
| `facial-ratios` | `facialRatios(frame)` | vertical thirds, horizontal fifths, fWHR, midface ratio |
| `eye-shape` | `eyeShape(frame)` | almond / round / narrow with the measurements behind it |
| `pupillary-distance` | `pupillaryDistance(points, size)` | PD estimate in mm, calibrated on the mean adult iris diameter (`IRIS_MM`) |

Lower-level pieces are exported too: `frame(points, aspect)` builds the
roll-corrected, pixel-consistent copy of the mesh that the metric functions
take; `gateFor(points, aspect, metric)` runs only the pose gate; `headPose`,
`faceBoxFromPoints`, `canthalTilt` and the `JAW` index list come from the same
code that guards photo quality in the app.

## Conventions

- Landmarks are MediaPipe's canonical mesh, normalized 0–1. 468 points are
  enough for most metrics; golden ratio, facial ratios and pupillary distance
  need the 478-point mesh with iris landmarks.
- "Left" and "right" are the subject's sides. The subject's right eye (index 33)
  sits on the image's left.
- Every measurement happens in pixel-consistent space (`x · aspect, y`) after
  removing head roll by rotating about the forehead-top (10) to menton (152)
  line, so the numbers do not change when the photo is tilted.
- Gates refuse rather than guess. Frontal-strict metrics (widths, symmetry,
  thirds/fifths) refuse a yaw beyond about 8°; eye metrics tolerate about 13°;
  pitch uses a wide band around neutral; a face under 15 % of the frame height
  is `small`; a short or non-finite mesh is `points`. A passing gate carries a
  `confidence` of `high` or `medium`.

## Try it in the browser

Each of these runs this library on-device; the photo never leaves the page.

- [Face shape](https://becometen.com/tools/face-shape)
- [Face symmetry](https://becometen.com/tools/face-symmetry)
- [Golden ratio](https://becometen.com/tools/golden-ratio)
- [Canthal tilt](https://becometen.com/tools/canthal-tilt)
- [Hunter eyes](https://becometen.com/tools/hunter-eyes)
- [Jawline score](https://becometen.com/tools/jawline-score)
- [Facial ratios](https://becometen.com/tools/facial-ratios)
- [Eye shape](https://becometen.com/tools/eye-shape)
- [Pupillary distance](https://becometen.com/tools/pupillary-distance)

## Not medical advice

These are geometric estimates from a single photo. They are educational, not a
diagnosis, and not a substitute for an optometrist, dermatologist or surgeon.

## Credits

Built for [BecomeTen](https://becometen.com), an AI facial-rating app whose
advice is gated on a PubMed-verified evidence library, by Appinara s. r. o.
MIT licensed.
