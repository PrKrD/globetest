import type { GridCell } from "../globe/GridLayer";

export class Tooltip {
  private el: HTMLElement;

  constructor(elementId: string) {
    const el = document.getElementById(elementId);
    if (!el) {
      throw new Error(`Tooltip element '${elementId}' not found`);
    }
    this.el = el;
  }

  show(cell: GridCell, x: number, y: number): void {
    const status = cell.is_land ? "land" : "ocean";
    const velocity = cell.has_velocity ? cell.velocity_speed_m_per_s!.toFixed(3) : "unavailable";
    const confidence = cell.equatorial_estimate
      ? "equatorial estimate — experimental"
      : cell.equatorial_uncertainty
        ? "equator — unavailable (Coriolis term is zero)"
        : "standard";
    this.el.innerHTML = `
      <div><strong>Cell:</strong> ${cell.id}</div>
      <div><strong>Lat:</strong> ${cell.clat.toFixed(3)}°</div>
      <div><strong>Lon:</strong> ${cell.clon.toFixed(3)}°</div>
      <div><strong>Status:</strong> ${status}</div>
      <div class="tooltip-section">
        <strong>Geostrophic velocity</strong>
        <div><strong>Speed:</strong> ${velocity}${cell.has_velocity ? " m/s" : ""}</div>
        <div><strong>Confidence:</strong> ${confidence}</div>
      </div>
    `;
    const margin = 10;
    const tooltipWidth = this.el.offsetWidth || 180;
    const tooltipHeight = this.el.offsetHeight || 84;
    const maxX = Math.max(margin, window.innerWidth - tooltipWidth - margin);
    const maxY = Math.max(margin, window.innerHeight - tooltipHeight - margin);
    const left = Math.min(Math.max(margin, x + 12), maxX);
    const top = Math.min(Math.max(margin, y + 12), maxY);
    this.el.style.left = `${left}px`;
    this.el.style.top = `${top}px`;
    this.el.classList.remove("hidden");
  }

  hide(): void {
    this.el.classList.add("hidden");
  }
}
