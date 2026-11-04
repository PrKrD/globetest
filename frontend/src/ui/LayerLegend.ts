import type { VisualizationMode } from "../globe/GridLayer";
import { legendStops } from "../globe/ColorScale";
import type { PercentileRange, ResolvedColorScale } from "../globe/ColorScale";

export class LayerLegend {
  private el: HTMLElement;
  private content: HTMLElement;
  private track: HTMLElement;
  private handles: HTMLButtonElement[];
  private fields: HTMLInputElement[];
  private range: PercentileRange = { lower: 2, upper: 98 };
  private enabled = false;
  private dragging: number | null = null;

  constructor(elementId: string, private onChange: (range: PercentileRange) => void) {
    const element = document.getElementById(elementId);
    if (!element) throw new Error(`Missing legend '${elementId}'`);
    this.el = element;
    this.el.innerHTML = `
      <div class="legend-content" aria-live="polite"></div>
      <div class="color-range">
        <div class="legend-title" id="colorRangeTitle">Color range (percentiles)</div>
        <div class="range-track" role="group" aria-labelledby="colorRangeTitle">
          <div class="range-selection"></div>
          <button type="button" class="range-handle" role="slider" aria-label="Lower percentile"></button>
          <button type="button" class="range-handle" role="slider" aria-label="Upper percentile"></button>
        </div>
        <div class="range-fields">
          <label>Lower <input type="number" aria-label="Lower percentile value" min="0" max="99" step="1"></label>
          <label>Upper <input type="number" aria-label="Upper percentile value" min="1" max="100" step="1"></label>
        </div>
      </div>`;
    this.content = this.el.querySelector<HTMLElement>(".legend-content")!;
    this.track = this.el.querySelector<HTMLElement>(".range-track")!;
    this.handles = Array.from(this.el.querySelectorAll<HTMLButtonElement>(".range-handle"));
    this.fields = Array.from(this.el.querySelectorAll<HTMLInputElement>("input"));
    this.fields.forEach((field, index) => {
      field.addEventListener("input", () => {
        if (field.value !== "" && Number.isFinite(field.valueAsNumber)) this.change(index, field.valueAsNumber);
      });
      field.addEventListener("change", () => this.syncControls(true));
      field.addEventListener("blur", () => this.syncControls(true));
    });
    this.handles.forEach((handle, index) => {
      handle.addEventListener("keydown", (event) => {
        const value = index === 0 ? this.range.lower : this.range.upper;
        const next = ({ ArrowLeft: value - 1, ArrowDown: value - 1,
          ArrowRight: value + 1, ArrowUp: value + 1, PageDown: value - 10,
          PageUp: value + 10, Home: index === 0 ? 0 : this.range.lower + 1,
          End: index === 0 ? this.range.upper - 1 : 100 } as Record<string, number>)[event.key];
        if (next === undefined) return;
        event.preventDefault();
        this.change(index, next);
      });
    });
    this.track.addEventListener("pointerdown", (event) => {
      if (!this.enabled || !event.isPrimary || event.button !== 0) return;
      const value = this.pointerValue(event);
      const index = this.handles.indexOf(event.target as HTMLButtonElement);
      this.dragging = index >= 0 ? index : Math.abs(value - this.range.lower) <= Math.abs(value - this.range.upper) ? 0 : 1;
      this.handles[this.dragging].focus();
      this.track.setPointerCapture(event.pointerId);
      event.preventDefault();
      this.change(this.dragging, value);
    });
    this.track.addEventListener("pointermove", (event) => {
      if (this.dragging !== null) this.change(this.dragging, this.pointerValue(event));
    });
    const endDrag = () => { this.dragging = null; };
    this.track.addEventListener("pointerup", endDrag);
    this.track.addEventListener("pointercancel", endDrag);
    this.track.addEventListener("lostpointercapture", endDrag);
  }

  private pointerValue(event: PointerEvent): number {
    const bounds = this.track.getBoundingClientRect();
    return Math.round((event.clientX - bounds.left) / bounds.width * 100);
  }

  private change(index: number, value: number): void {
    if (!this.enabled) return;
    const next = Math.round(value);
    this.range = index === 0
      ? { ...this.range, lower: Math.max(0, Math.min(next, this.range.upper - 1)) }
      : { ...this.range, upper: Math.max(this.range.lower + 1, Math.min(next, 100)) };
    this.syncControls();
    this.onChange({ ...this.range });
  }

  private syncControls(commit = false): void {
    const values = [this.range.lower, this.range.upper];
    this.handles.forEach((handle, index) => {
      const minimum = index === 0 ? 0 : this.range.lower + 1;
      const maximum = index === 0 ? this.range.upper - 1 : 100;
      handle.style.left = `${values[index]}%`;
      handle.setAttribute("aria-valuemin", String(minimum));
      handle.setAttribute("aria-valuemax", String(maximum));
      handle.setAttribute("aria-valuenow", String(values[index]));
      handle.setAttribute("aria-valuetext", `Percentile ${values[index]}`);
      handle.disabled = !this.enabled;
      this.fields[index].disabled = !this.enabled;
      this.fields[index].min = String(minimum);
      this.fields[index].max = String(maximum);
      if (commit || document.activeElement !== this.fields[index]) this.fields[index].value = String(values[index]);
    });
    const selection = this.el.querySelector<HTMLElement>(".range-selection")!;
    selection.style.left = `${this.range.lower}%`;
    selection.style.width = `${this.range.upper - this.range.lower}%`;
  }

  show(mode: VisualizationMode, scale: ResolvedColorScale | null, range: PercentileRange): void {
    this.el.classList.toggle("hidden", mode === "none");
    this.range = { ...range };
    this.enabled = scale !== null && mode !== "none";
    this.syncControls();
    if (mode === "none") return;
    const title = mode === "mld" ? "Maximum MLD H (m)" : mode === "velocity" ? "Geostrophic speed (m/s)" : "Geostrophic subduction (m/s)";
    if (!scale) {
      this.content.innerHTML = `<div class="legend-title">${title}</div><div class="legend-note">No valid ocean values available</div>`;
      return;
    }
    const format = (value: number) => value === 0 ? "0" : Math.abs(value) < 0.01 || Math.abs(value) >= 10000
      ? value.toExponential(2) : Number(value.toPrecision(4)).toString();
    const middle = scale.minimum < 0 && scale.maximum > 0 ? 0 : (scale.minimum + scale.maximum) / 2;
    const position = scale.minimum === scale.maximum ? 50 : 100 * (middle - scale.minimum) / (scale.maximum - scale.minimum);
    const constant = scale.minimum === scale.maximum;
    this.content.innerHTML = `
      <div class="legend-title">${title}</div>
      <div class="legend-bar" style="background:linear-gradient(90deg, ${legendStops(scale)})"></div>
      <div class="legend-labels"><span>≤${format(scale.minimum)}</span><span>≥${format(scale.maximum)}</span></div>
      ${!constant && position >= 20 && position <= 80 ? `<div class="legend-middle" style="left:${position}%">${format(middle)}</div>` : ""}
      <div class="legend-note">p${range.lower}–p${range.upper} · ${constant ? "Constant value" : mode === "subduction" ? "Linear signed scale; neutral = zero" : "Square-root scale"}</div>
      ${mode === "mld" ? "" : '<div class="legend-note">Red = unavailable; equator switch off also cautions the experimental band</div>'}`;
  }
}
