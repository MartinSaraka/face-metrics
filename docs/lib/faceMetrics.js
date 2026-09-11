import { canthalTilt, JAW } from "./geometry.js";
import { faceBoxFromPoints, headPose, PITCH_MAX, PITCH_MIN, PITCH_NEUTRAL, } from "./geometry.js";
export const TOOL_METRICS = [
    "face-shape",
    "face-symmetry",
    "golden-ratio",
    "canthal-tilt",
    "hunter-eyes",
    "jawline-score",
    "facial-ratios",
    "eye-shape",
    "pupillary-distance",
];
// ——— Landmark anchors ———
const FOREHEAD_TOP = 10;
const GLABELLA = 9;
const SUBNASALE = 2;
const MENTON = 152;
const UPPER_LIP_TOP = 0;
const STOMION = 13;
const FOREHEAD_W = [54, 284];
const BIZYGOMATIC = [234, 454];
const BIGONIAL = [172, 397];
const CHIN_W = [176, 400];
const ALAE = [129, 358];
const MOUTH = [61, 291];
// Gonial angle: at the gonion, between the ray up the ramus and the ray
// along the mandible toward the chin.
const GONION_R = { at: 172, up: 132, chin: 150 };
const GONION_L = { at: 397, up: 361, chin: 379 };
const EYE_R = {
    outer: 33,
    inner: 133,
    upper: 159,
    lower: 145,
    brow: 105,
    pupil: 468,
    rimH: [469, 471],
    rimV: [470, 472],
};
const EYE_L = {
    outer: 263,
    inner: 362,
    upper: 386,
    lower: 374,
    brow: 334,
    pupil: 473,
    rimH: [474, 476],
    rimV: [475, 477],
};
const MESH_POINTS = 468;
const IRIS_POINTS = 478;
// ——— Gate thresholds ———
// photoQuality.headPose defines yaw as (noseTip.x − eyeMidX)/faceWidth — a
// ratio of face width, where 0.12 ≈ a 15–20° turn. The tools are stricter than
// the photo check because they measure single traits that a turn distorts:
// frontal-strict metrics (widths, symmetry, thirds/fifths) refuse |yaw| > 0.06
// (≈ 8°), eye metrics tolerate 0.10 (≈ 13°). Roll is removed before measuring,
// so it only refuses when the rotation itself is suspect (> 15° off the eye
// line). Pitch uses the photo check's band (0.3–0.9 around a 0.5 neutral).
// "high" confidence means every pose number sits inside half its threshold
// and the face fills at least twice the minimum height.
export const YAW_STRICT = 0.06;
export const YAW_EYES = 0.1;
export const ROLL_MAX = 15;
export const MIN_FACE_HEIGHT = 0.15;
const EYE_METRICS = new Set([
    "canthal-tilt",
    "hunter-eyes",
    "eye-shape",
    "pupillary-distance",
]);
const IRIS_METRICS = new Set([
    "golden-ratio",
    "facial-ratios",
    "pupillary-distance",
]);
export function gateFor(points, aspect, metric) {
    const need = IRIS_METRICS.has(metric) ? IRIS_POINTS : MESH_POINTS;
    if (!Array.isArray(points) || points.length < need) {
        return { ok: false, reason: "points" };
    }
    const pose = headPose(points, aspect);
    if (!pose)
        return { ok: false, reason: "points" };
    const box = faceBoxFromPoints(points);
    if (!box)
        return { ok: false, reason: "points" };
    const yawMax = EYE_METRICS.has(metric) ? YAW_EYES : YAW_STRICT;
    if (Math.abs(pose.yawRatio) > yawMax)
        return { ok: false, reason: "yaw" };
    if (pose.pitchRatio < PITCH_MIN || pose.pitchRatio > PITCH_MAX) {
        return { ok: false, reason: "pitch" };
    }
    if (Math.abs(pose.rollDeg) > ROLL_MAX)
        return { ok: false, reason: "roll" };
    if (box.h < MIN_FACE_HEIGHT)
        return { ok: false, reason: "small" };
    const pitchHigh = pose.pitchRatio >= PITCH_NEUTRAL - (PITCH_NEUTRAL - PITCH_MIN) / 2 &&
        pose.pitchRatio <= PITCH_NEUTRAL + (PITCH_MAX - PITCH_NEUTRAL) / 2;
    const high = Math.abs(pose.yawRatio) <= yawMax / 2 &&
        Math.abs(pose.rollDeg) <= ROLL_MAX / 2 &&
        pitchHigh &&
        box.h >= MIN_FACE_HEIGHT * 2;
    return { ok: true, confidence: high ? "high" : "medium" };
}
function clamp01(n) {
    return Math.min(1, Math.max(0, n));
}
function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}
function mid(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}
// Linear map clamped to 0–100: `lo` → 0, `hi` → 100 (either order).
function ramp(v, lo, hi) {
    return 100 * clamp01((v - lo) / (hi - lo));
}
/**
 * Roll-corrected, pixel-consistent copy of the mesh: x scaled by the image
 * aspect, then rotated about the midpoint of 10 and 152 until that line is
 * vertical. Exported for tests; metric functions take its output.
 */
export function frame(points, aspect) {
    const px = points.map((p) => ({ x: p.x * aspect, y: p.y }));
    const top = px[FOREHEAD_TOP];
    const bottom = px[MENTON];
    if (!top || !bottom)
        return px;
    const c = mid(top, bottom);
    const vx = bottom.x - top.x;
    const vy = bottom.y - top.y;
    if (Math.hypot(vx, vy) < 1e-9)
        return px;
    // Angle of the midline from the +y axis; rotate by −θ to make it vertical.
    const theta = Math.atan2(vx, vy);
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    return px.map((p) => {
        const dx = p.x - c.x;
        const dy = p.y - c.y;
        return { x: c.x + dx * cos - dy * sin, y: c.y + dx * sin + dy * cos };
    });
}
// Interior angle at `at` between the rays to `a` and `b`, degrees.
function angleAt(at, a, b) {
    const ax = a.x - at.x;
    const ay = a.y - at.y;
    const bx = b.x - at.x;
    const by = b.y - at.y;
    const den = Math.hypot(ax, ay) * Math.hypot(bx, by);
    if (den < 1e-12)
        return 0;
    const cos = Math.max(-1, Math.min(1, (ax * bx + ay * by) / den));
    return (Math.acos(cos) * 180) / Math.PI;
}
function span(f, [a, b]) {
    return dist(f[a], f[b]);
}
export const SHAPE_TOLERANCE = [
    0.12, 0.06, 0.06, 0.08, 10,
];
export const SHAPE_PROTOTYPES = [
    { shape: "oval", features: [1.45, 0.92, 0.85, 0.5, 125] },
    { shape: "round", features: [1.25, 0.9, 0.95, 0.62, 135] },
    { shape: "square", features: [1.3, 0.95, 1.0, 0.7, 115] },
    { shape: "oblong", features: [1.65, 0.92, 0.9, 0.55, 125] },
    { shape: "heart", features: [1.4, 1.02, 0.78, 0.36, 125] },
    { shape: "diamond", features: [1.45, 0.8, 0.8, 0.4, 125] },
];
function gonialAngles(f) {
    return {
        right: angleAt(f[GONION_R.at], f[GONION_R.up], f[GONION_R.chin]),
        left: angleAt(f[GONION_L.at], f[GONION_L.up], f[GONION_L.chin]),
    };
}
export function faceShape(f) {
    const bizygomatic = span(f, BIZYGOMATIC);
    const bigonial = span(f, BIGONIAL);
    const g = gonialAngles(f);
    const feat = [
        dist(f[FOREHEAD_TOP], f[MENTON]) / bizygomatic,
        span(f, FOREHEAD_W) / bizygomatic,
        bigonial / bizygomatic,
        span(f, CHIN_W) / bigonial,
        (g.left + g.right) / 2,
    ];
    const scored = SHAPE_PROTOTYPES.map((p) => {
        let d = 0;
        for (let i = 0; i < 5; i++) {
            const z = (feat[i] - p.features[i]) / SHAPE_TOLERANCE[i];
            d += z * z;
        }
        return { shape: p.shape, d: Math.sqrt(d) };
    }).sort((a, b) => a.d - b.d);
    const best = scored[0];
    const next = scored[1];
    const match = next.d > 0 ? clamp01(1 - best.d / next.d) : 1;
    return {
        shape: best.shape,
        match,
        runnerUp: next.shape,
        ratios: {
            length: feat[0],
            forehead: feat[1],
            jaw: feat[2],
            chin: feat[3],
        },
        gonial: feat[4],
    };
}
// ——— Symmetry ———
// Pairs as [subject-right (image left), subject-left (image right)].
export const SYMMETRY_GROUPS = {
    eyes: [
        [33, 263],
        [133, 362],
        [159, 386],
        [145, 374],
    ],
    brows: [
        [70, 300],
        [63, 293],
        [105, 334],
        [66, 296],
        [107, 336],
    ],
    nose: [
        [129, 358],
        [48, 278],
    ],
    mouth: [
        [61, 291],
        [78, 308],
    ],
    // Face contour: jaw, cheeks and forehead/temple oval points.
    jaw: [
        [234, 454],
        [93, 323],
        [132, 361],
        [58, 288],
        [172, 397],
        [136, 365],
        [150, 379],
        [176, 400],
        [50, 280],
        [123, 352],
        [116, 345],
        [54, 284],
        [103, 332],
        [67, 297],
        [109, 338],
        [21, 251],
        [162, 389],
        [127, 356],
    ],
};
export const SYMMETRY_PAIRS = Object.values(SYMMETRY_GROUPS).flat();
// Mean pair deviation (as a fraction of face width) at which the score hits 0.
const SYMMETRY_FLOOR = 0.08;
function symmetryScore(f, x0, width, pairs) {
    let sum = 0;
    for (const [r, l] of pairs) {
        const dx = f[l].x - x0 - (x0 - f[r].x);
        const dy = f[l].y - f[r].y;
        sum += Math.hypot(dx, dy) / width;
    }
    return 100 * clamp01(1 - sum / pairs.length / SYMMETRY_FLOOR);
}
export function symmetry(f) {
    const x0 = (f[FOREHEAD_TOP].x + f[MENTON].x) / 2;
    const width = span(f, BIZYGOMATIC);
    const part = (k) => symmetryScore(f, x0, width, SYMMETRY_GROUPS[k]);
    return {
        score: symmetryScore(f, x0, width, SYMMETRY_PAIRS),
        parts: {
            eyes: part("eyes"),
            brows: part("brows"),
            nose: part("nose"),
            mouth: part("mouth"),
            jaw: part("jaw"),
        },
    };
}
// ——— Golden ratio ———
const PHI = 1.618;
const PHI_FLOOR = Math.log(1.5); // |ln(r/φ)| at which a ratio scores 0
function ipd(f) {
    return dist(f[EYE_R.pupil], f[EYE_L.pupil]);
}
export function goldenRatio(f) {
    const mouth = span(f, MOUTH);
    const raw = [
        [
            "Face length to width",
            dist(f[FOREHEAD_TOP], f[MENTON]) / span(f, BIZYGOMATIC),
        ],
        ["Mouth width to nose width", mouth / span(f, ALAE)],
        [
            "Nose to chin over nose to lips",
            (f[MENTON].y - f[SUBNASALE].y) / (f[STOMION].y - f[SUBNASALE].y),
        ],
        ["Pupil distance to mouth width", ipd(f) / mouth],
    ];
    const ratios = raw.map(([name, value]) => ({
        name,
        value,
        target: PHI,
        score: value > 0 && Number.isFinite(value)
            ? 100 * clamp01(1 - Math.abs(Math.log(value / PHI)) / PHI_FLOOR)
            : 0,
    }));
    return {
        score: ratios.reduce((a, r) => a + r.score, 0) / ratios.length,
        ratios,
    };
}
function eyeStats(f, e) {
    const outer = f[e.outer];
    const inner = f[e.inner];
    const width = dist(outer, inner);
    const height = dist(f[e.upper], f[e.lower]);
    return {
        width,
        ar: height / width,
        browGap: (f[e.upper].y - f[e.brow].y) / width,
        tilt: canthalTilt(inner, outer, 1),
        curve: (f[e.lower].y - mid(outer, inner).y) / width,
    };
}
export function canthalTiltMetric(f) {
    const right = eyeStats(f, EYE_R).tilt;
    const left = eyeStats(f, EYE_L).tilt;
    return { left, right, mean: (left + right) / 2 };
}
export function hunterEyes(f) {
    const r = eyeStats(f, EYE_R);
    const l = eyeStats(f, EYE_L);
    const ar = (r.ar + l.ar) / 2;
    const browGap = (r.browGap + l.browGap) / 2;
    const tilt = (r.tilt + l.tilt) / 2;
    const score = 0.45 * ramp(ar, 0.42, 0.24) +
        0.35 * ramp(browGap, 0.75, 0.35) +
        0.2 * ramp(tilt, -2, 8);
    return { score, ar, browGap, tilt };
}
export function eyeShape(f) {
    const r = eyeStats(f, EYE_R);
    const l = eyeStats(f, EYE_L);
    const ar = (r.ar + l.ar) / 2;
    const tilt = (r.tilt + l.tilt) / 2;
    const curve = (r.curve + l.curve) / 2;
    return {
        shape: ar > 0.36 ? "round" : ar < 0.28 ? "narrow" : "almond",
        modifier: tilt > 2 ? "upturned" : tilt < -2 ? "downturned" : "level",
        ar,
        tilt,
        curve,
    };
}
// ——— Jawline ———
export function jawline(f) {
    const g = gonialAngles(f);
    const gonialMean = (g.left + g.right) / 2;
    const bigonial = span(f, BIGONIAL);
    const chinTaper = span(f, CHIN_W) / bigonial;
    const jawToCheek = bigonial / span(f, BIZYGOMATIC);
    // Angle: a defined 108–126° corner scores full; 0 by 90° (too boxy) or
    // 150° (no corner at all). Width: 0.80–0.95 of the cheekbones is full; 0
    // at 0.60 or 1.10.
    const angleScore = gonialMean < 108
        ? ramp(gonialMean, 90, 108)
        : gonialMean > 126
            ? ramp(gonialMean, 150, 126)
            : 100;
    const widthScore = jawToCheek < 0.8
        ? ramp(jawToCheek, 0.6, 0.8)
        : jawToCheek > 0.95
            ? ramp(jawToCheek, 1.1, 0.95)
            : 100;
    return {
        score: 0.6 * angleScore + 0.4 * widthScore,
        gonialLeft: g.left,
        gonialRight: g.right,
        gonialMean,
        chinTaper,
        jawToCheek,
    };
}
// ——— Facial ratios ———
export function facialRatios(f) {
    const ys = [FOREHEAD_TOP, GLABELLA, SUBNASALE, MENTON].map((i) => f[i].y);
    const total = ys[3] - ys[0];
    const thirds = [
        ((ys[1] - ys[0]) / total) * 100,
        ((ys[2] - ys[1]) / total) * 100,
        ((ys[3] - ys[2]) / total) * 100,
    ];
    const xs = [234, 33, 133, 362, 263, 454].map((i) => f[i].x);
    const width = xs[5] - xs[0];
    const fifths = [0, 1, 2, 3, 4].map((i) => ((xs[i + 1] - xs[i]) / width) * 100);
    const lipTop = f[UPPER_LIP_TOP].y;
    const lidY = (f[EYE_R.upper].y + f[EYE_L.upper].y) / 2;
    const pupilY = (f[EYE_R.pupil].y + f[EYE_L.pupil].y) / 2;
    return {
        thirds,
        fifths,
        fwhr: span(f, BIZYGOMATIC) / (lipTop - lidY),
        midface: ipd(f) / (lipTop - pupilY),
    };
}
// ——— Pupillary distance ———
// Mean adult iris diameter; near-constant across people, so it calibrates the
// scale of an uncalibrated photo to ±~7 %.
export const IRIS_MM = 11.7;
export function pupillaryDistance(points, size) {
    // Real pixels, not the roll-corrected frame: rotation preserves distances
    // and the ratio is scale-free anyway.
    const px = (i) => ({
        x: points[i].x * size.w,
        y: points[i].y * size.h,
    });
    const ipdPx = dist(px(EYE_R.pupil), px(EYE_L.pupil));
    const rims = [
        EYE_R.rimH,
        EYE_R.rimV,
        EYE_L.rimH,
        EYE_L.rimV,
    ];
    const irisPx = rims.reduce((a, [p, q]) => a + dist(px(p), px(q)), 0) / rims.length;
    const mm = (ipdPx * IRIS_MM) / irisPx;
    return { px: ipdPx, mm, range: [mm * 0.93, mm * 1.07] };
}
// ——— Overlays ———
const EYE_LINES = [
    [EYE_R.inner, EYE_R.outer],
    [EYE_L.inner, EYE_L.outer],
];
function overlayFor(metric) {
    switch (metric) {
        case "face-shape":
            return {
                polylines: [JAW],
                lines: [FOREHEAD_W, BIZYGOMATIC, BIGONIAL, CHIN_W],
                points: [FOREHEAD_TOP, MENTON],
            };
        case "face-symmetry":
            return {
                vlines: [FOREHEAD_TOP],
                points: SYMMETRY_PAIRS.flat(),
            };
        case "golden-ratio":
            return {
                lines: [
                    [FOREHEAD_TOP, MENTON],
                    BIZYGOMATIC,
                    MOUTH,
                    ALAE,
                    [EYE_R.pupil, EYE_L.pupil],
                ],
                hlines: [SUBNASALE, STOMION],
            };
        case "canthal-tilt":
            return {
                lines: EYE_LINES,
                points: [EYE_R.inner, EYE_R.outer, EYE_L.inner, EYE_L.outer],
            };
        case "hunter-eyes":
            return {
                lines: [
                    ...EYE_LINES,
                    [EYE_R.upper, EYE_R.lower],
                    [EYE_L.upper, EYE_L.lower],
                    [EYE_R.brow, EYE_R.upper],
                    [EYE_L.brow, EYE_L.upper],
                ],
            };
        case "jawline-score":
            return {
                polylines: [JAW],
                lines: [
                    [GONION_R.up, GONION_R.at],
                    [GONION_R.at, GONION_R.chin],
                    [GONION_L.up, GONION_L.at],
                    [GONION_L.at, GONION_L.chin],
                    BIGONIAL,
                ],
                points: [GONION_R.at, GONION_L.at],
            };
        case "facial-ratios":
            return {
                hlines: [FOREHEAD_TOP, GLABELLA, SUBNASALE, MENTON],
                vlines: [234, 33, 133, 362, 263, 454],
            };
        case "eye-shape":
            return {
                lines: [
                    ...EYE_LINES,
                    [EYE_R.upper, EYE_R.lower],
                    [EYE_L.upper, EYE_L.lower],
                ],
                points: [EYE_R.lower, EYE_L.lower],
            };
        case "pupillary-distance":
            return {
                lines: [[EYE_R.pupil, EYE_L.pupil], EYE_R.rimH, EYE_L.rimH],
                points: [EYE_R.pupil, EYE_L.pupil],
            };
    }
}
// ——— Dispatcher ———
function allFinite(v) {
    if (typeof v === "number")
        return Number.isFinite(v);
    if (typeof v === "string" || typeof v === "boolean")
        return true;
    if (Array.isArray(v))
        return v.every(allFinite);
    if (v && typeof v === "object")
        return Object.values(v).every(allFinite);
    return false;
}
function compute(metric, points, aspect, size) {
    const f = frame(points, aspect);
    const table = {
        "face-shape": () => faceShape(f),
        "face-symmetry": () => symmetry(f),
        "golden-ratio": () => goldenRatio(f),
        "canthal-tilt": () => canthalTiltMetric(f),
        "hunter-eyes": () => hunterEyes(f),
        "jawline-score": () => jawline(f),
        "facial-ratios": () => facialRatios(f),
        "eye-shape": () => eyeShape(f),
        "pupillary-distance": () => pupillaryDistance(points, size),
    };
    return table[metric]();
}
/**
 * Gate + measure in one call. `aspect` is image width/height; `size` (natural
 * pixel size) is only needed for the pixel figure of pupillary distance and
 * overrides `aspect` when given. Never throws: any non-finite result becomes
 * a "points" refusal with a null value.
 */
export function measure(metric, points, aspect, size) {
    const overlay = overlayFor(metric);
    const a = size && size.w > 0 && size.h > 0
        ? size.w / size.h
        : Number.isFinite(aspect) && aspect > 0
            ? aspect
            : 1;
    const gate = gateFor(points, a, metric);
    if (!gate.ok)
        return { gate, value: null, overlay };
    try {
        const value = compute(metric, points, a, size ?? { w: a, h: 1 });
        if (!allFinite(value)) {
            return { gate: { ok: false, reason: "points" }, value: null, overlay };
        }
        return { gate, value, overlay };
    }
    catch {
        return { gate: { ok: false, reason: "points" }, value: null, overlay };
    }
}
