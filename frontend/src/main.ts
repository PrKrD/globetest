import "cesium/Build/Cesium/Widgets/widgets.css";
import { EllipsoidTerrainProvider, OpenStreetMapImageryProvider, Viewer } from "cesium";

import { FRONTEND_GRID_DATA_FALLBACK_URL, FRONTEND_GRID_MANIFEST_URL } from "./config";
import { GridLayer } from "./globe/GridLayer";
import { GridPicking } from "./globe/GridPicking";
import { HoverDeformation } from "./globe/HoverDeformation";
import { Tooltip } from "./ui/Tooltip";
import "./style.css";

interface GridManifest {
  coarse_grid_path: string;
  coarse_resolution_deg: number;
}

async function resolveGridDataUrl(): Promise<string> {
  try {
    const response = await fetch(FRONTEND_GRID_MANIFEST_URL);
    if (!response.ok) {
      return FRONTEND_GRID_DATA_FALLBACK_URL;
    }
    const manifest = (await response.json()) as GridManifest;
    if (!manifest.coarse_grid_path) {
      return FRONTEND_GRID_DATA_FALLBACK_URL;
    }
    return manifest.coarse_grid_path;
  } catch (_error) {
    return FRONTEND_GRID_DATA_FALLBACK_URL;
  }
}

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
  await gridLayer.loadFromUrl(await resolveGridDataUrl());

  const tooltip = new Tooltip("tooltip");
  const deformation = new HoverDeformation(viewer, gridLayer.getDataset());

  const picking = new GridPicking(viewer, gridLayer, (cell, x, y) => {
    deformation.updateHoveredCell(cell);
    if (!cell) {
      tooltip.hide();
      return;
    }
    tooltip.show(cell, x, y);
  });

  window.addEventListener("beforeunload", () => {
    picking.destroy();
    deformation.destroy();
  });
}

bootstrap().catch((error) => {
  console.error(error);
});
