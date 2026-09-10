// Geometry helpers the metrics depend on, extracted verbatim from BecomeTen
// (faceGeometry.ts / photoQuality.ts). Landmarks are MediaPipe's 478-point
// face mesh, normalized 0–1 in image space.

export type Pt = { x: number; y: number };

export type Landmark = { x: number; y: number; z?: number };

export const LANDMARK_COUNT = 468; // 478 with iris; both accepted

// Jawline, ear to ear via the chin (subject-right → chin → subject-left).
export const JAW = [
	234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
	397, 288, 361, 323, 454,
];

// Angle of the canthal line in degrees, positive when the OUTER canthus sits
// higher than the inner one (image y grows downward). x is scaled by the
// image aspect so the angle is measured in pixel space, not normalized space.
// (Pass aspect 1 for points already in pixel space.)
export function canthalTilt(inner: Pt, outer: Pt, aspect: number): number {
	const dx = Math.abs(outer.x - inner.x) * aspect;
	const dy = inner.y - outer.y;
	if (dx === 0) return 0;
	return (Math.atan2(dy, dx) * 180) / Math.PI;
}

export type PoseMetrics = {
	// Angle of the eye line vs horizontal, in degrees, measured in pixel space
	// (positive = subject-left eye lower in the image).
	rollDeg: number;
	// Nose-tip horizontal offset from the eye midpoint / face width. 0 frontal.
	yawRatio: number;
	// (noseTip→chin) / (foreheadTop→noseTip) vertical proportion. ~0.5 neutral.
	pitchRatio: number;
};

// Roll beyond this (degrees) skews the canthal-tilt measurement.
export const ROLL_MAX_DEG = 8;
// |yawRatio| beyond 0.12 corresponds to a head turned roughly 15–20° — far
// enough that one jaw side is foreshortened.
export const YAW_MAX_RATIO = 0.12;
// On a frontal MediaPipe mesh the nose tip sits about two thirds of the way
// down the forehead-top(10)→chin(152) span, so neutral pitchRatio ≈ 0.5.
// Tipping the head back (chin up) foreshortens forehead→nose and grows the
// ratio; chin down shrinks it. The band is wide (roughly ±20° of pitch).
export const PITCH_NEUTRAL = 0.5;
export const PITCH_MIN = 0.3; // below → chinDown
export const PITCH_MAX = 0.9; // above → chinUp

// Mesh indices (MediaPipe canonical mesh).
const NOSE_TIP = 1;
const FOREHEAD_TOP = 10;
const CHIN = 152;
const EYE_OUTER_R = 33; // subject-right outer canthus (image left)
const EYE_OUTER_L = 263; // subject-left outer canthus (image right)
const FACE_OVAL_R = 234; // widest oval point, subject right
const FACE_OVAL_L = 454; // widest oval point, subject left

const MIN_POINTS = 468; // 478 with iris; both accepted

function finitePt(points: Landmark[], i: number): Landmark | null {
	const p = points[i];
	if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) return null;
	return p;
}

/**
 * Head pose from the landmark set. `aspect` is image w/h, used to measure the
 * roll angle in pixel space rather than normalized space. Returns null on
 * short/degenerate input; never throws.
 */
export function headPose(
	points: Landmark[],
	aspect: number,
): PoseMetrics | null {
	if (!Array.isArray(points) || points.length < MIN_POINTS) return null;
	if (!Number.isFinite(aspect) || aspect <= 0) return null;
	const nose = finitePt(points, NOSE_TIP);
	const forehead = finitePt(points, FOREHEAD_TOP);
	const chin = finitePt(points, CHIN);
	const outerR = finitePt(points, EYE_OUTER_R);
	const outerL = finitePt(points, EYE_OUTER_L);
	const ovalR = finitePt(points, FACE_OVAL_R);
	const ovalL = finitePt(points, FACE_OVAL_L);
	if (!nose || !forehead || !chin || !outerR || !outerL || !ovalR || !ovalL) {
		return null;
	}

	// Roll: angle of the outer-canthus line. Normalized dy over aspect-scaled
	// dx equals the pixel-space angle (dyPix/dxPix = dyN·h / dxN·w).
	const dx = (outerL.x - outerR.x) * aspect;
	const dy = outerL.y - outerR.y;
	if (Math.abs(dx) < 1e-6) return null;
	const rollDeg = (Math.atan2(dy, dx) * 180) / Math.PI;

	// Yaw: on a frontal face the nose tip projects onto the eye midpoint; a
	// turned head shifts it toward the far cheek. Normalized by face width so
	// distance to the camera cancels out.
	const faceWidth = Math.abs(ovalL.x - ovalR.x);
	if (faceWidth < 1e-3) return null;
	const eyeMidX = (outerR.x + outerL.x) / 2;
	const yawRatio = (nose.x - eyeMidX) / faceWidth;

	// Pitch: vertical proportion of the face around the nose tip. See the
	// PITCH_* constants for the neutral baseline and direction.
	const upper = nose.y - forehead.y;
	const lower = chin.y - nose.y;
	if (upper < 1e-3 || lower < 0) return null;
	const pitchRatio = lower / upper;

	return { rollDeg, yawRatio, pitchRatio };
}

/**
 * Bounding box (normalized 0–1) of all finite landmarks.
 */
export function faceBoxFromPoints(
	points: Landmark[],
): { x: number; y: number; w: number; h: number } | null {
	let x0 = Number.POSITIVE_INFINITY;
	let y0 = Number.POSITIVE_INFINITY;
	let x1 = Number.NEGATIVE_INFINITY;
	let y1 = Number.NEGATIVE_INFINITY;
	for (const p of points) {
		if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
		const x = Math.min(1, Math.max(0, p.x));
		const y = Math.min(1, Math.max(0, p.y));
		x0 = Math.min(x0, x);
		y0 = Math.min(y0, y);
		x1 = Math.max(x1, x);
		y1 = Math.max(y1, y);
	}
	if (!Number.isFinite(x0) || x1 - x0 < 1e-3 || y1 - y0 < 1e-3) return null;
	return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}
