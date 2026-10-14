export type VelocityVisibility = { enabled: boolean; equatorialEnabled: boolean };

export class VelocityControls {
  private master: HTMLInputElement;
  private equatorial: HTMLInputElement;

  constructor(onChange: (state: VelocityVisibility) => void) {
    this.master = this.control("velocityToggle");
    this.equatorial = this.control("equatorialToggle");
    const sync = () => {
      this.equatorial.disabled = !this.master.checked;
      onChange({ enabled: this.master.checked, equatorialEnabled: this.equatorial.checked });
    };
    this.master.addEventListener("change", sync);
    this.equatorial.addEventListener("change", sync);
    sync();
  }

  private control(id: string): HTMLInputElement {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLInputElement)) throw new Error(`Missing control '${id}'`);
    return element;
  }
}
