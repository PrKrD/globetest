import {
  Cartesian2,
  Cartesian3,
  Color,
  ColorGeometryInstanceAttribute,
  GeometryInstance,
  PerInstanceColorAppearance,
  PolygonGeometry,
  PolygonHierarchy,
  Primitive,
  Scene,
  Viewer,
  defined,
} from "cesium";

import { LAND_COLOR, OCEAN_COLOR } from "../config";

export interface GridCell {
  id: number;
  lat0: number;
  lat1: number;
  lon0: number;
  lon1: number;
  clat: number;
  clon: number;
  is_land: number;
}

interface CoarseGridPayload {
  resolution_deg: number;
  n_lat: number;
  n_lon: number;
  cells: GridCell[];
}

export class GridLayer {
  private scene: Scene;
  private landPrimitive: Primitive | undefined;
  private oceanPrimitive: Primitive | undefined;

  constructor(viewer: Viewer) {
    this.scene = viewer.scene;
  }

  async loadFromUrl(url: string): Promise<void> {
    const response = await fetch(url);
    const payload = (await response.json()) as CoarseGridPayload;

    const landInstances: GeometryInstance[] = [];
    const oceanInstances: GeometryInstance[] = [];

    for (const cell of payload.cells) {
      const instance = this.buildCellInstance(cell);
      if (cell.is_land) {
        landInstances.push(instance);
      } else {
        oceanInstances.push(instance);
      }
    }

    this.landPrimitive = this.buildPrimitive(landInstances, LAND_COLOR);
    this.oceanPrimitive = this.buildPrimitive(oceanInstances, OCEAN_COLOR);

    this.scene.primitives.add(this.landPrimitive);
    this.scene.primitives.add(this.oceanPrimitive);
  }

  pickCell(windowPosition: { x: number; y: number }): GridCell | null {
    const picked = this.scene.pick(new Cartesian2(windowPosition.x, windowPosition.y));
    if (!defined(picked) || !("id" in picked) || !picked.id) {
      return null;
    }
    return picked.id as GridCell;
  }

  private buildPrimitive(instances: GeometryInstance[], hexColor: string): Primitive {
    const color = Color.fromCssColorString(hexColor).withAlpha(0.55);
    for (const instance of instances) {
      instance.attributes = {
        color: ColorGeometryInstanceAttribute.fromColor(color),
      };
    }

    return new Primitive({
      geometryInstances: instances,
      appearance: new PerInstanceColorAppearance({
        translucent: true,
        closed: false,
      }),
      asynchronous: true,
    });
  }

  private buildCellInstance(cell: GridCell): GeometryInstance {
    const lat0 = cell.lat0 <= -90 ? -89.999 : cell.lat0;
    const lat1 = cell.lat1 >= 90 ? 89.999 : cell.lat1;
    const positions = Cartesian3.fromDegreesArray([
      cell.lon0,
      lat0,
      cell.lon1,
      lat0,
      cell.lon1,
      lat1,
      cell.lon0,
      lat1,
    ]);

    return new GeometryInstance({
      geometry: new PolygonGeometry({
        polygonHierarchy: new PolygonHierarchy(positions),
      }),
      id: cell,
    });
  }
}
