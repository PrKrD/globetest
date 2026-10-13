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
  has_velocity?: number;
  velocity_speed_m_per_s?: number;
  equatorial_uncertainty?: number;
}

export interface GridDataset {
  resolution_deg: number;
  n_lat: number;
  n_lon: number;
  dataset_type?: string;
  velocity_p98_m_per_s?: number;
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
  private showVelocity = true;

  constructor(viewer: Viewer) {
    this.scene = viewer.scene;
  }

  async loadFromUrl(url: string): Promise<void> {
    const response = await fetch(url);
    const payload = (await response.json()) as GridDataset;
    this.dataset = this.withIndices(payload);
    this.cellsById = new Map(this.dataset.cells.map((cell) => [cell.id, cell]));

    this.rebuildDataPrimitives();
    if (!this.dataset.dataset_type) {
      this.boundaryPrimitive = this.buildBoundaryPrimitive(this.dataset);
      this.scene.primitives.add(this.boundaryPrimitive);
    }
  }

  setVelocityVisualization(enabled: boolean): void {
    this.showVelocity = enabled;
    this.rebuildDataPrimitives();
  }

  private rebuildDataPrimitives(): void {
    if (!this.dataset) return;
    if (this.landPrimitive) this.scene.primitives.remove(this.landPrimitive);
    if (this.oceanPrimitive) this.scene.primitives.remove(this.oceanPrimitive);
    const landInstances: GeometryInstance[] = [];
    const oceanInstances: GeometryInstance[] = [];

    for (const cell of this.dataset.cells) {
      const instance = this.buildCellInstance(cell);
      if (cell.is_land) {
        instance.attributes = { color: ColorGeometryInstanceAttribute.fromColor(Color.fromCssColorString(LAND_COLOR)) };
        landInstances.push(instance);
      } else {
        instance.attributes = {
          color: ColorGeometryInstanceAttribute.fromColor(this.colorForCell(cell)),
        };
        oceanInstances.push(instance);
      }
    }

    this.landPrimitive = this.buildPrimitive(landInstances, LAND_COLOR);
    this.oceanPrimitive = this.buildPrimitive(oceanInstances, OCEAN_COLOR);

    this.scene.primitives.add(this.landPrimitive);
    this.scene.primitives.add(this.oceanPrimitive);
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
      instance.attributes ??= { color: ColorGeometryInstanceAttribute.fromColor(color) };
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

  private colorForCell(cell: GridCell): Color {
    if (!this.showVelocity) {
      return Color.fromCssColorString(OCEAN_COLOR);
    }
    if (cell.equatorial_uncertainty) return Color.fromCssColorString("#d94a43");
    if (!cell.has_velocity || cell.velocity_speed_m_per_s === undefined || !this.dataset?.velocity_p98_m_per_s) return Color.fromCssColorString(OCEAN_COLOR);
    const normalized = Math.sqrt(Math.min(cell.velocity_speed_m_per_s / this.dataset.velocity_p98_m_per_s, 1));
    const low = Color.fromCssColorString("#123b7a");
    const middle = Color.fromCssColorString("#24b6b0");
    const high = Color.fromCssColorString("#f2d13d");
    return normalized < 0.55
      ? Color.lerp(low, middle, normalized / 0.55, new Color())
      : Color.lerp(middle, high, (normalized - 0.55) / 0.45, new Color());
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
