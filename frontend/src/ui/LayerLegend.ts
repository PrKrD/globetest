import type { GridDataset, VisualizationMode } from "../globe/GridLayer";

export class LayerLegend {
  private el: HTMLElement;

  constructor(elementId: string) {
    const element = document.getElementById(elementId);
    if (!element) throw new Error(`Missing legend '${elementId}'`);
    this.el = element;
  }

  show(mode: VisualizationMode, dataset: GridDataset): void {
    if (mode === "none") {
      this.el.classList.add("hidden");
      return;
    }

    const isMld = mode === "mld";
    const maximum = isMld ? dataset.mld_max_p98_m : dataset.velocity_p98_m_per_s;
    if (maximum === undefined) {
      this.el.classList.add("hidden");
      return;
    }
    const middle = isMld ? maximum * 0.25 : maximum * 0.55 ** 2;
    const digits = isMld ? 0 : 2;
    this.el.innerHTML = `
      <div class="legend-title">${isMld ? "Maximum MLD H (m)" : "Geostrophic speed (m/s)"}</div>
      <div class="legend-bar ${isMld ? "legend-mld" : "legend-velocity"}"></div>
      <div class="legend-labels">
        <span>0</span><span>${middle.toFixed(digits)}</span><span>≥${maximum.toFixed(digits)}</span>
      </div>
      <div class="legend-note">Square-root scale; upper limit is p98</div>
    `;
    this.el.classList.remove("hidden");
  }
}
