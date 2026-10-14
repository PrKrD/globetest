import type { VisualizationMode } from "../globe/GridLayer";

export type LayerVisibility = {
  mode: VisualizationMode;
  equatorialEnabled: boolean;
  subductionEquatorialEnabled: boolean;
};

export class LayerControls {
  private subduction: HTMLInputElement;
  private velocity: HTMLInputElement;
  private mld: HTMLInputElement;
  private equatorial: HTMLInputElement;
  private subductionEquatorial: HTMLInputElement;

  constructor(onChange: (state: LayerVisibility) => void) {
    this.subduction = this.control("subductionToggle");
    this.velocity = this.control("velocityToggle");
    this.mld = this.control("mldToggle");
    this.equatorial = this.control("equatorialToggle");
    this.subductionEquatorial = this.control("subductionEquatorialToggle");

    this.subduction.addEventListener("change", () => {
      if (this.subduction.checked) {
        this.velocity.checked = false;
        this.mld.checked = false;
      }
      this.sync(onChange);
    });
    this.velocity.addEventListener("change", () => {
      if (this.velocity.checked) {
        this.subduction.checked = false;
        this.mld.checked = false;
      }
      this.sync(onChange);
    });
    this.mld.addEventListener("change", () => {
      if (this.mld.checked) {
        this.subduction.checked = false;
        this.velocity.checked = false;
      }
      this.sync(onChange);
    });
    this.equatorial.addEventListener("change", () => this.sync(onChange));
    this.subductionEquatorial.addEventListener("change", () => this.sync(onChange));
    this.sync(onChange);
  }

  private sync(onChange: (state: LayerVisibility) => void): void {
    this.equatorial.disabled = !this.velocity.checked;
    this.subductionEquatorial.disabled = !this.subduction.checked;
    const mode: VisualizationMode = this.subduction.checked
      ? "subduction"
      : this.mld.checked
        ? "mld"
        : this.velocity.checked
          ? "velocity"
          : "none";
    onChange({
      mode,
      equatorialEnabled: this.equatorial.checked,
      subductionEquatorialEnabled: this.subductionEquatorial.checked,
    });
  }

  private control(id: string): HTMLInputElement {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLInputElement)) throw new Error(`Missing control '${id}'`);
    return element;
  }
}
