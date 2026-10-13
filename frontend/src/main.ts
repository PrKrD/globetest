import "cesium/Build/Cesium/Widgets/widgets.css";
import { EllipsoidTerrainProvider, OpenStreetMapImageryProvider, Viewer } from "cesium";

import { FRONTEND_DYNTOPO_DATA_URL } from "./config";
import { GridLayer } from "./globe/GridLayer";
import { GridPicking } from "./globe/GridPicking";
import { Tooltip } from "./ui/Tooltip";
import { VelocityControls } from "./ui/VelocityControls";
import "./style.css";

async function bootstrap(): Promise<void> {
  const viewer = new Viewer("cesiumContainer", {
    animation: false,
    timeline: false,
    geocoder: false,
    homeButton: false,
    navigationHelpButton: false,
    sceneModePicker: false,
    baseLayerPicker: false,
    fullscreenButton: false,
    infoBox: false,
    selectionIndicator: false,
    baseLayer: false,
    terrainProvider: new EllipsoidTerrainProvider(),
  });
  viewer.imageryLayers.addImageryProvider(
    new OpenStreetMapImageryProvider({
      url: "https://tile.openstreetmap.org/",
    }),
  );

  const gridLayer = new GridLayer(viewer);
  await gridLayer.loadFromUrl(FRONTEND_DYNTOPO_DATA_URL);

  const tooltip = new Tooltip("tooltip");
  new VelocityControls((state) => {
    gridLayer.setVelocityVisualization(state.enabled);
  });

  const picking = new GridPicking(viewer, gridLayer, (cell, x, y) => {
    if (!cell) {
      tooltip.hide();
      return;
    }
    tooltip.show(cell, x, y);
  });

  window.addEventListener("beforeunload", () => {
    picking.destroy();
  });
}

bootstrap().catch((error) => {
  console.error(error);
});
