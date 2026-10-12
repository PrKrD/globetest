import { ScreenSpaceEventHandler, ScreenSpaceEventType, Viewer } from "cesium";

import { GridCell, GridLayer } from "./GridLayer";

export class GridPicking {
  private handler: ScreenSpaceEventHandler;

  constructor(
    viewer: Viewer,
    gridLayer: GridLayer,
    onHover: (cell: GridCell | null, x: number, y: number) => void,
  ) {
    this.handler = new ScreenSpaceEventHandler(viewer.canvas);
    this.handler.setInputAction((movement: { endPosition: { x: number; y: number } }) => {
      const cell = gridLayer.pickCell(movement.endPosition);
      onHover(cell, movement.endPosition.x, movement.endPosition.y);
    }, ScreenSpaceEventType.MOUSE_MOVE);
    this.handler.setInputAction((movement: { position: { x: number; y: number } }) => {
      const cell = gridLayer.pickCell(movement.position);
      onHover(cell, movement.position.x, movement.position.y);
    }, ScreenSpaceEventType.LEFT_CLICK);
  }

  destroy(): void {
    this.handler.destroy();
  }
}
