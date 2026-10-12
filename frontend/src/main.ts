import "cesium/Build/Cesium/Widgets/widgets.css";
import { Viewer } from "cesium";

import { FRONTEND_GRID_DATA_URL } from "./config";
import { GridLayer } from "./globe/GridLayer";
import { GridPicking } from "./globe/GridPicking";
import { HoverDeformation } from "./globe/HoverDeformation";
import { Tooltip } from "./ui/Tooltip";
import "./style.css";

async function bootstrap(): Promise<void> {
  const viewer = new Viewer("cesiumContainer", {
    animation: false,
    timeline: false,
    geocoder: false,
    homeButton: true,
    navigationHelpButton: true,
    sceneModePicker: false,
    baseLayerPicker: false,
    infoBox: false,
    selectionIndicator: false,
  });

  const gridLayer = new GridLayer(viewer);
  await gridLayer.loadFromUrl(FRONTEND_GRID_DATA_URL);

  const tooltip = new Tooltip("tooltip");
  const deformation = new HoverDeformation();

  new GridPicking(viewer, gridLayer, (cell, x, y) => {
    deformation.updateHoveredCell(cell);
    if (!cell) {
      tooltip.hide();
      return;
    }
    tooltip.show(cell, x, y);
  });
}

bootstrap().catch((error) => {
  console.error(error);
});
