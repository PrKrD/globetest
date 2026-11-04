import type { GridCell } from "./GridLayer";

export type DataMode = "velocity" | "mld" | "subduction";
export type PercentileRange = { lower: number; upper: number };
export type ResolvedColorScale = PercentileRange & {
  mode: DataMode;
  minimum: number;
  maximum: number;
};
export type RGB = readonly [number, number, number];
export const defaultRange = (): PercentileRange => ({ lower: 2, upper: 98 });
const palettes: Record<DataMode, readonly [RGB, RGB, RGB]> = {
  velocity: [[18, 59, 122], [36, 182, 176], [242, 209, 61]],
  mld: [[68, 1, 84], [33, 145, 140], [253, 231, 37]],
  subduction: [[94, 60, 153], [255, 255, 191], [27, 158, 119]],
};
const clamp = (value: number, lower: number, upper: number) => Math.max(lower, Math.min(value, upper));

export function cellValue(cell: GridCell, mode: DataMode): number | undefined {
  if (cell.is_land) return undefined;
  const value = mode === "velocity"
    ? cell.has_velocity ? cell.velocity_speed_m_per_s : undefined
    : mode === "mld"
      ? cell.has_mld ? cell.mld_max_m : undefined
      : cell.has_subduction ? cell.subduction_m_per_s : undefined;
  return value !== undefined && Number.isFinite(value) ? value : undefined;
}

export function sortedValues(cells: GridCell[], mode: DataMode): number[] {
  return cells.flatMap((cell) => {
    const value = cellValue(cell, mode);
    return value === undefined ? [] : [value];
  }).sort((a, b) => a - b);
}

export function quantile(sorted: number[], percentile: number): number | undefined {
  if (!sorted.length) return undefined;
  const index = (sorted.length - 1) * clamp(percentile, 0, 100) / 100;
  const lower = Math.floor(index);
  return sorted[lower] + (sorted[Math.ceil(index)] - sorted[lower]) * (index - lower);
}

export function resolveScale(mode: DataMode, sorted: number[], range: PercentileRange): ResolvedColorScale | null {
  const minimum = quantile(sorted, range.lower);
  const maximum = quantile(sorted, range.upper);
  return minimum === undefined || maximum === undefined ? null : { mode, ...range, minimum, maximum };
}

// Signed positions preserve zero at the palette midpoint, even for one-sign ranges.
export function normalize(value: number, scale: ResolvedColorScale): number {
  const { minimum, maximum, mode } = scale;
  const clipped = clamp(value, minimum, maximum);
  if (mode === "subduction") {
    if (clipped < 0) return 0.5 * (1 - clipped / minimum);
    if (clipped > 0) return 0.5 + 0.5 * clipped / maximum;
    return 0.5;
  }
  return minimum === maximum ? 0.5 : Math.sqrt((clipped - minimum) / (maximum - minimum));
}

export function colorAt(value: number, scale: ResolvedColorScale): RGB {
  const t = normalize(value, scale);
  const pivot = scale.mode === "velocity" ? 0.55 : 0.5;
  const palette = palettes[scale.mode];
  const [a, b, fraction] = t < pivot
    ? [palette[0], palette[1], t / pivot] as const
    : [palette[1], palette[2], (t - pivot) / (1 - pivot)] as const;
  const channel = (i: number) => Math.round(a[i] + (b[i] - a[i]) * fraction);
  return [channel(0), channel(1), channel(2)];
}

export function legendStops(scale: ResolvedColorScale): string {
  // The bar is linear in physical values; sample the same transfer function as cells.
  const positions = Array.from({ length: 101 }, (_, i) => i / 100);
  if (scale.minimum < 0 && scale.maximum > 0) {
    positions.push(-scale.minimum / (scale.maximum - scale.minimum));
    positions.sort((a, b) => a - b);
  }
  return positions.map((position) => {
    const color = colorAt(scale.minimum + (scale.maximum - scale.minimum) * position, scale);
    return `rgb(${color.join(",")}) ${position * 100}%`;
  }).join(", ");
}
