import { describe, expect, it } from "vitest";
import type { Landmark } from "./geometry";
import {
	faceShape,
	frame,
	gateFor,
	measure,
	SHAPE_PROTOTYPES,
	symmetry,
} from "./faceMetrics";

// A schematic, perfectly symmetric face in a square image: every point starts
// at the face centre, then the anchors the metrics read are placed from a
// handful of proportions. Enough to pin sign conventions, the classifier and
// the calibration without a real mesh.
type Proportions = {
	length: number; // face length / bizygomatic
	forehead: number; // forehead width / bizygomatic
	jaw: number; // bigonial / bizygomatic
	chin: number; // chin width / bigonial
	gonial: number; // gonial angle, degrees
	ipd?: number; // pupil distance, normalized
};

const W = 0.4; // bizygomatic width
const TOP = 0.2; // forehead top y

function build(p: Proportions): Landmark[] {
	const pts: Landmark[] = Array.from({ length: 478 }, () => ({
		x: 0.5,
		y: 0.45,
	}));
	const set = (i: number, x: number, y: number) => {
		pts[i] = { x, y };
	};
	const len = p.length * W;
	const y = (frac: number) => TOP + len * frac;
	set(10, 0.5, y(0));
	set(9, 0.5, y(0.33));
	set(1, 0.5, y(2 / 3)); // nose tip → pitchRatio 0.5
	set(2, 0.5, y(0.63));
	set(0, 0.5, y(0.75));
	set(13, 0.5, y(0.8));
	set(152, 0.5, y(1));
	set(234, 0.5 - W / 2, y(0.5));
	set(454, 0.5 + W / 2, y(0.5));
	set(54, 0.5 - (p.forehead * W) / 2, y(0.12));
	set(284, 0.5 + (p.forehead * W) / 2, y(0.12));
	const gx = (p.jaw * W) / 2;
	const gy = y(0.8);
	set(172, 0.5 - gx, gy);
	set(397, 0.5 + gx, gy);
	set(176, 0.5 - p.chin * gx, y(0.97));
	set(400, 0.5 + p.chin * gx, y(0.97));
	// Ramus straight up from the gonion; mandible ray rotated `gonial` degrees
	// from it toward the chin.
	const rad = (p.gonial * Math.PI) / 180;
	set(132, 0.5 - gx, gy - 0.08);
	set(361, 0.5 + gx, gy - 0.08);
	set(150, 0.5 - gx + 0.08 * Math.sin(rad), gy - 0.08 * Math.cos(rad));
	set(379, 0.5 + gx - 0.08 * Math.sin(rad), gy - 0.08 * Math.cos(rad));
	// Eyes, level, aspect ratio 0.3.
	const ey = y(0.4);
	const half = (p.ipd ?? 0.2) / 2;
	for (const [sign, e] of [
		[
			-1,
			{
				outer: 33,
				inner: 133,
				up: 159,
				lo: 145,
				brow: 105,
				pupil: 468,
				rim: 469,
			},
		],
		[
			1,
			{
				outer: 263,
				inner: 362,
				up: 386,
				lo: 374,
				brow: 334,
				pupil: 473,
				rim: 474,
			},
		],
	] as const) {
		set(e.outer, 0.5 + sign * 0.14, ey);
		set(e.inner, 0.5 + sign * 0.06, ey);
		set(e.up, 0.5 + sign * 0.1, ey - 0.012);
		set(e.lo, 0.5 + sign * 0.1, ey + 0.012);
		set(e.brow, 0.5 + sign * 0.1, ey - 0.05);
		const px = 0.5 + sign * half;
		set(e.pupil, px, ey);
		set(e.rim, px - 0.01, ey);
		set(e.rim + 1, px, ey - 0.01);
		set(e.rim + 2, px + 0.01, ey);
		set(e.rim + 3, px, ey + 0.01);
	}
	set(129, 0.45, y(0.62));
	set(358, 0.55, y(0.62));
	set(61, 0.41, y(0.8));
	set(291, 0.59, y(0.8));
	return pts;
}

const OVAL: Proportions = {
	length: 1.45,
	forehead: 0.92,
	jaw: 0.85,
	chin: 0.5,
	gonial: 125,
};

function rotated(points: Landmark[], deg: number): Landmark[] {
	const rad = (deg * Math.PI) / 180;
	const c = Math.cos(rad);
	const s = Math.sin(rad);
	return points.map((p) => {
		const dx = p.x - 0.5;
		const dy = p.y - 0.5;
		return { x: 0.5 + dx * c - dy * s, y: 0.5 + dx * s + dy * c };
	});
}

describe("faceMetrics", () => {
	it("scores a mirror-symmetric face 100 and notices a shifted canthus", () => {
		const base = build(OVAL);
		const r = measure("face-symmetry", base, 1);
		expect(r.gate.ok).toBe(true);
		expect(r.value?.score).toBeCloseTo(100, 5);
		const shifted = build(OVAL);
		shifted[33] = { x: shifted[33].x + 0.02, y: shifted[33].y };
		const s = measure("face-symmetry", shifted, 1);
		expect(s.value).not.toBeNull();
		expect(s.value?.score ?? 100).toBeLessThan(100);
		expect(s.value?.parts.eyes ?? 100).toBeLessThan(100);
		expect(s.value?.parts.mouth).toBeCloseTo(100, 5);
	});

	it("classifies every prototype as itself", () => {
		for (const p of SHAPE_PROTOTYPES) {
			const [length, forehead, jaw, chin, gonial] = p.features;
			const pts = build({ length, forehead, jaw, chin, gonial });
			const v = faceShape(frame(pts, 1));
			expect(v.shape, p.shape).toBe(p.shape);
			expect(v.match).toBeGreaterThan(0.5);
			expect(v.gonial).toBeCloseTo(gonial, 3);
			const m = measure("face-shape", pts, 1);
			expect(m.gate).toEqual({ ok: true, confidence: "high" });
			expect(m.value?.shape).toBe(p.shape);
		}
	});

	it("signs canthal tilt positive when the outer corner is higher", () => {
		const level = measure("canthal-tilt", build(OVAL), 1);
		expect(level.value?.mean).toBeCloseTo(0, 6);
		const up = build(OVAL);
		up[33] = { x: up[33].x, y: up[33].y - 0.01 };
		up[263] = { x: up[263].x, y: up[263].y - 0.01 };
		const v = measure("canthal-tilt", up, 1).value;
		expect(v?.left ?? 0).toBeGreaterThan(0);
		expect(v?.right ?? 0).toBeGreaterThan(0);
		expect(v?.mean).toBeCloseTo(Math.atan2(0.01, 0.08) * (180 / Math.PI), 4);
		const down = build(OVAL);
		down[33] = { x: down[33].x, y: down[33].y + 0.01 };
		down[263] = { x: down[263].x, y: down[263].y + 0.01 };
		expect(measure("canthal-tilt", down, 1).value?.mean ?? 0).toBeLessThan(0);
	});

	it("measures a right-angle jaw as 90 degrees", () => {
		const r = measure("jawline-score", build({ ...OVAL, gonial: 90 }), 1);
		expect(r.value?.gonialLeft).toBeCloseTo(90, 4);
		expect(r.value?.gonialRight).toBeCloseTo(90, 4);
		expect(r.value?.gonialMean).toBeCloseTo(90, 4);
	});

	it("calibrates pupillary distance from the iris diameter", () => {
		// Iris span 0.02, pupils 0.11 apart, 1000×1000 image → 110 px × 11.7 / 20 px.
		const r = measure("pupillary-distance", build({ ...OVAL, ipd: 0.11 }), 1, {
			w: 1000,
			h: 1000,
		});
		expect(r.gate.ok).toBe(true);
		expect(r.value?.px).toBeCloseTo(110, 6);
		expect(r.value?.mm ?? 0).toBeGreaterThan(63);
		expect(r.value?.mm ?? 0).toBeLessThan(65.5);
		expect(r.value?.range[0] ?? 0).toBeLessThan(r.value?.mm ?? 0);
		expect(r.value?.range[1] ?? 0).toBeGreaterThan(r.value?.mm ?? 0);
	});

	it("refuses short input and a turned head", () => {
		const short = build(OVAL).slice(0, 400);
		expect(gateFor(short, 1, "face-shape")).toEqual({
			ok: false,
			reason: "points",
		});
		expect(measure("face-shape", short, 1).value).toBeNull();
		// Nose tip a quarter face-width off the eye midline → yawRatio 0.25.
		const yawed = build(OVAL);
		yawed[1] = { x: 0.6, y: yawed[1].y };
		expect(gateFor(yawed, 1, "face-symmetry")).toEqual({
			ok: false,
			reason: "yaw",
		});
		expect(gateFor(yawed, 1, "canthal-tilt")).toEqual({
			ok: false,
			reason: "yaw",
		});
		// A mild turn passes the eye metrics but not the frontal-strict ones.
		const mild = build(OVAL);
		mild[1] = { x: 0.532, y: mild[1].y }; // yawRatio 0.08
		expect(gateFor(mild, 1, "face-shape").ok).toBe(false);
		expect(gateFor(mild, 1, "canthal-tilt").ok).toBe(true);
	});

	it("is invariant to a small roll", () => {
		const straight = symmetry(frame(build(OVAL), 1)).score;
		const tilted = build(OVAL);
		tilted[33] = { x: tilted[33].x + 0.02, y: tilted[33].y };
		const a = symmetry(frame(tilted, 1)).score;
		const b = symmetry(frame(rotated(tilted, 5), 1)).score;
		expect(a).toBeLessThan(straight);
		expect(Math.abs(a - b)).toBeLessThan(1);
		const m = measure("face-symmetry", rotated(build(OVAL), 5), 1);
		expect(m.gate.ok).toBe(true);
		expect(m.value?.score ?? 0).toBeGreaterThan(99);
	});

	it("reports the thirds and fifths as percentages that sum to 100", () => {
		const r = measure("facial-ratios", build(OVAL), 1);
		const t = r.value?.thirds ?? [0, 0, 0];
		const f = r.value?.fifths ?? [0, 0, 0, 0, 0];
		expect(t[0] + t[1] + t[2]).toBeCloseTo(100, 6);
		expect(f.reduce((a, n) => a + n, 0)).toBeCloseTo(100, 6);
		expect(r.value?.fwhr ?? 0).toBeGreaterThan(1);
	});

	it("never throws on junk coordinates", () => {
		const junk = build(OVAL);
		junk[33] = { x: Number.NaN, y: 2 };
		for (const metric of [
			"face-shape",
			"face-symmetry",
			"golden-ratio",
			"canthal-tilt",
			"hunter-eyes",
			"jawline-score",
			"facial-ratios",
			"eye-shape",
			"pupillary-distance",
		] as const) {
			const r = measure(metric, junk, 1);
			expect(r.gate.ok).toBe(false);
			expect(r.value).toBeNull();
		}
		expect(measure("eye-shape", [], Number.NaN).value).toBeNull();
	});
});
