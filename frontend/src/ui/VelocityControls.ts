export type VelocityVisibility = { enabled: boolean };

export class VelocityControls {
  private master: HTMLInputElement;

  constructor(onChange: (state: VelocityVisibility) => void) {
    this.master = this.control("velocityToggle");
    const sync = () => onChange({ enabled: this.master.checked });
    this.master.addEventListener("change", sync);
    sync();
  }

  private control(id: string): HTMLInputElement {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLInputElement)) throw new Error(`Missing control '${id}'`);
    return element;
  }
}
