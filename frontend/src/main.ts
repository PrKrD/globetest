import "cesium/Build/Cesium/Widgets/widgets.css";
import { EllipsoidTerrainProvider, OpenStreetMapImageryProvider, Viewer } from "cesium";

import { FRONTEND_DYNTOPO_DATA_URL } from "./config";
import { GridLayer } from "./globe/GridLayer";
import { GridPicking } from "./globe/GridPicking";
import { Tooltip } from "./ui/Tooltip";
import { LayerControls } from "./ui/LayerControls";
import { LayerLegend } from "./ui/LayerLegend";
import { defaultRange } from "./globe/ColorScale";
import type { DataMode, PercentileRange } from "./globe/ColorScale";
import type { LayerVisibility } from "./ui/LayerControls";
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
  const ranges: Record<DataMode, PercentileRange> = {
    velocity: defaultRange(), mld: defaultRange(), subduction: defaultRange(),
  };
  let state: LayerVisibility = { mode: "mld", equatorialEnabled: true, subductionEquatorialEnabled: true };
  let pendingFrame: number | null = null;
  const render = () => {
    pendingFrame = null;
    const range = state.mode === "none" ? defaultRange() : ranges[state.mode];
    gridLayer.setVisualization(state.mode, state.equatorialEnabled, state.subductionEquatorialEnabled, range);
    legend.show(state.mode, gridLayer.getColorScale(), range);
  };
  const legend = new LayerLegend("layerLegend", (range) => {
    if (state.mode === "none") return;
    ranges[state.mode] = range;
    if (pendingFrame === null) pendingFrame = requestAnimationFrame(render);
  });
  new LayerControls((nextState) => {
    state = nextState;
    if (pendingFrame !== null) cancelAnimationFrame(pendingFrame);
    render();
  });

  const picking = new GridPicking(viewer, gridLayer, (cell, x, y) => {
    if (!cell) {
      tooltip.hide();
      return;
    }
    tooltip.show(cell, x, y);
  });

  window.addEventListener("beforeunload", () => {
    if (pendingFrame !== null) cancelAnimationFrame(pendingFrame);
    picking.destroy();
  });
}

bootstrap().catch((error) => {
  console.error(error);
});
