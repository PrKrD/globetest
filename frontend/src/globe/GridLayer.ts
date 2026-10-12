import {
  ArcType,
  Cartesian2,
  Cartesian3,
  Color,
  ColorGeometryInstanceAttribute,
  GeometryInstance,
  PerInstanceColorAppearance,
  PolylineColorAppearance,
  PolylineGeometry,
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
  i: number;
  j: number;
  lat0: number;
  lat1: number;
  lon0: number;
  lon1: number;
  clat: number;
  clon: number;
  is_land: number;
}

export interface GridDataset {
  resolution_deg: number;
  n_lat: number;
  n_lon: number;
  cells: GridCell[];
}

const BASE_CELL_HEIGHT_METERS = 5.0;
const GRID_LINE_HEIGHT_METERS = 8.0;

export class GridLayer {
  private scene: Scene;
  private landPrimitive: Primitive | undefined;
  private oceanPrimitive: Primitive | undefined;
  private boundaryPrimitive: Primitive | undefined;
  private dataset: GridDataset | null = null;
  private cellsById = new Map<number, GridCell>();

  constructor(viewer: Viewer) {
    this.scene = viewer.scene;
  }

  async loadFromUrl(url: string): Promise<void> {
    const response = await fetch(url);
    const payload = (await response.json()) as GridDataset;
    this.dataset = this.withIndices(payload);
    this.cellsById = new Map(this.dataset.cells.map((cell) => [cell.id, cell]));

    const landInstances: GeometryInstance[] = [];
    const oceanInstances: GeometryInstance[] = [];

    for (const cell of this.dataset.cells) {
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
    this.boundaryPrimitive = this.buildBoundaryPrimitive(this.dataset);
    this.scene.primitives.add(this.boundaryPrimitive);
  }

  getDataset(): GridDataset {
    if (!this.dataset) {
      throw new Error("Grid dataset is not loaded");
    }
    return this.dataset;
  }

  getCellById(cellId: number): GridCell | null {
    return this.cellsById.get(cellId) ?? null;
  }

  pickCell(windowPosition: { x: number; y: number }): GridCell | null {
    const picked = this.scene.pick(new Cartesian2(windowPosition.x, windowPosition.y));
    if (!defined(picked) || !("id" in picked) || !picked.id) {
      return null;
    }
    const pickedCell = picked.id as GridCell;
    return this.getCellById(pickedCell.id);
  }

  private buildPrimitive(instances: GeometryInstance[], hexColor: string): Primitive {
    const color = Color.fromCssColorString(hexColor).withAlpha(0.88);
    for (const instance of instances) {
      instance.attributes = {
        color: ColorGeometryInstanceAttribute.fromColor(color),
      };
    }

    return new Primitive({
      geometryInstances: instances,
      appearance: new PerInstanceColorAppearance({
        translucent: false,
        closed: false,
        flat: true,
        renderState: {
          depthTest: { enabled: true },
          depthMask: true,
        },
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
        height: BASE_CELL_HEIGHT_METERS,
        vertexFormat: PerInstanceColorAppearance.FLAT_VERTEX_FORMAT,
      }),
      id: cell,
    });
  }

  private withIndices(payload: GridDataset): GridDataset {
    const { n_lon } = payload;
    return {
      ...payload,
      cells: payload.cells.map((cell) => ({
        ...cell,
        i: Math.floor(cell.id / n_lon),
        j: cell.id % n_lon,
      })),
    };
  }

  private buildBoundaryPrimitive(dataset: GridDataset): Primitive {
    const instances: GeometryInstance[] = [];
    const step = dataset.resolution_deg;
    const lineColor = ColorGeometryInstanceAttribute.fromColor(
      Color.fromCssColorString("#dce6f6").withAlpha(0.22),
    );

    for (let lon = -180; lon <= 180 + 1e-9; lon += step) {
      const degrees: number[] = [];
      for (let lat = -90; lat <= 90 + 1e-9; lat += step) {
        const safeLat = Math.max(-89.999, Math.min(89.999, lat));
        degrees.push(lon, safeLat);
      }
      instances.push(
        new GeometryInstance({
          geometry: new PolylineGeometry({
            positions: Cartesian3.fromDegreesArrayHeights(
              this.withHeights(degrees, GRID_LINE_HEIGHT_METERS),
            ),
            width: 1.0,
            vertexFormat: PolylineColorAppearance.VERTEX_FORMAT,
            arcType: ArcType.GEODESIC,
          }),
          attributes: { color: lineColor },
        }),
      );
    }

    for (let lat = -90; lat <= 90 + 1e-9; lat += step) {
      const safeLat = Math.max(-89.999, Math.min(89.999, lat));
      const degrees: number[] = [];
      for (let lon = -180; lon <= 180 + 1e-9; lon += step) {
        degrees.push(lon, safeLat);
      }
      instances.push(
        new GeometryInstance({
          geometry: new PolylineGeometry({
            positions: Cartesian3.fromDegreesArrayHeights(
              this.withHeights(degrees, GRID_LINE_HEIGHT_METERS),
            ),
            width: 1.0,
            vertexFormat: PolylineColorAppearance.VERTEX_FORMAT,
            arcType: ArcType.GEODESIC,
          }),
          attributes: { color: lineColor },
        }),
      );
    }

    return new Primitive({
      geometryInstances: instances,
      appearance: new PolylineColorAppearance({
        translucent: true,
      }),
      asynchronous: true,
      allowPicking: false,
      releaseGeometryInstances: true,
    });
  }

  private withHeights(degrees: number[], heightMeters: number): number[] {
    const out: number[] = [];
    for (let idx = 0; idx < degrees.length; idx += 2) {
      out.push(degrees[idx], degrees[idx + 1], heightMeters);
    }
    return out;
  }
}
