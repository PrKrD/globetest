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
    this.el.innerHTML = `
      <div><strong>Cell:</strong> ${cell.id}</div>
      <div><strong>Lat:</strong> ${cell.clat.toFixed(3)}°</div>
      <div><strong>Lon:</strong> ${cell.clon.toFixed(3)}°</div>
      <div><strong>Status:</strong> ${status}</div>
    `;
    this.el.style.left = `${x + 12}px`;
    this.el.style.top = `${y + 12}px`;
    this.el.classList.remove("hidden");
  }

  hide(): void {
    this.el.classList.add("hidden");
  }
}
