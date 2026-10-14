import type { VisualizationMode } from "../globe/GridLayer";

export type LayerVisibility = {
  mode: VisualizationMode;
  equatorialEnabled: boolean;
};

export class LayerControls {
  private velocity: HTMLInputElement;
  private mld: HTMLInputElement;
  private equatorial: HTMLInputElement;

  constructor(onChange: (state: LayerVisibility) => void) {
    this.velocity = this.control("velocityToggle");
    this.mld = this.control("mldToggle");
    this.equatorial = this.control("equatorialToggle");

    this.velocity.addEventListener("change", () => {
      if (this.velocity.checked) this.mld.checked = false;
      this.sync(onChange);
    });
    this.mld.addEventListener("change", () => {
      if (this.mld.checked) this.velocity.checked = false;
      this.sync(onChange);
    });
    this.equatorial.addEventListener("change", () => this.sync(onChange));
    this.sync(onChange);
  }

  private sync(onChange: (state: LayerVisibility) => void): void {
    this.equatorial.disabled = !this.velocity.checked;
    const mode: VisualizationMode = this.velocity.checked
      ? "velocity"
      : this.mld.checked
        ? "mld"
        : "none";
    onChange({ mode, equatorialEnabled: this.equatorial.checked });
  }

  private control(id: string): HTMLInputElement {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLInputElement)) throw new Error(`Missing control '${id}'`);
    return element;
  }
}
