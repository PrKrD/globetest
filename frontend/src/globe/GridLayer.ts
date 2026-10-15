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
  equatorial_estimate?: number;
  has_mld?: number;
  mld_max_m?: number;
  mld_month_count_min?: number;
  has_subduction?: number;
  subduction_m_per_s?: number;
}

export interface GridDataset {
  resolution_deg: number;
  n_lat: number;
  n_lon: number;
  dataset_type?: string;
  velocity_p98_m_per_s?: number;
  mld_max_p98_m?: number;
  subduction_abs_p98_m_per_s?: number;
  cells: GridCell[];
}

export type VisualizationMode = "none" | "velocity" | "mld" | "subduction";

const BASE_CELL_HEIGHT_METERS = 5.0;
const GRID_LINE_HEIGHT_METERS = 8.0;
const LAND_CELL_COLOR = Color.fromCssColorString(LAND_COLOR);
const OCEAN_CELL_COLOR = Color.fromCssColorString(OCEAN_COLOR);
const EQUATORIAL_CAUTION_COLOR = Color.fromCssColorString("#d94a43");
const VELOCITY_LOW_COLOR = Color.fromCssColorString("#123b7a");
const VELOCITY_MIDDLE_COLOR = Color.fromCssColorString("#24b6b0");
const VELOCITY_HIGH_COLOR = Color.fromCssColorString("#f2d13d");
const MLD_LOW_COLOR = Color.fromCssColorString("#440154");
const MLD_MIDDLE_COLOR = Color.fromCssColorString("#21918c");
const MLD_HIGH_COLOR = Color.fromCssColorString("#fde725");
const SUBDUCTION_NEGATIVE_COLOR = Color.fromCssColorString("#5e3c99");
const SUBDUCTION_ZERO_COLOR = Color.fromCssColorString("#ffffbf");
const SUBDUCTION_POSITIVE_COLOR = Color.fromCssColorString("#1b9e77");

export class GridLayer {
  private scene: Scene;
  private dataPrimitive: Primitive | undefined;
  private boundaryPrimitive: Primitive | undefined;
  private dataset: GridDataset | null = null;
  private cellsById = new Map<number, GridCell>();
  private visualizationMode: VisualizationMode = "mld";
  private showEquatorialEstimates = true;
  private showEquatorialSubduction = true;

  constructor(viewer: Viewer) {
    this.scene = viewer.scene;
  }

  async loadFromUrl(url: string): Promise<void> {
    const response = await fetch(url);
    const payload = (await response.json()) as GridDataset;
    this.dataset = this.withIndices(payload);
    this.cellsById = new Map(this.dataset.cells.map((cell) => [cell.id, cell]));

    this.buildDataPrimitives();
    if (!this.dataset.dataset_type) {
      this.boundaryPrimitive = this.buildBoundaryPrimitive(this.dataset);
      this.scene.primitives.add(this.boundaryPrimitive);
    }
  }

  setVisualization(
    mode: VisualizationMode,
    equatorialEnabled: boolean,
    subductionEquatorialEnabled: boolean,
  ): void {
    this.visualizationMode = mode;
    this.showEquatorialEstimates = equatorialEnabled;
    this.showEquatorialSubduction = subductionEquatorialEnabled;
    this.updateCellColors();
  }

  private buildDataPrimitives(): void {
    if (!this.dataset) return;
    const instances = this.dataset.cells.map((cell) =>
      this.buildColoredCellInstance(cell, this.colorForCell(cell)),
    );
    this.dataPrimitive = this.buildPrimitive(instances);
    this.scene.primitives.add(this.dataPrimitive);

    // Per-instance attributes are available only after Cesium has prepared the
    // primitive. Apply the latest selected mode then in case it changed while
    // the asynchronous geometry build was in progress.
    const removeReadyListener = this.scene.postRender.addEventListener(() => {
      if (!this.dataPrimitive?.ready) return;
      removeReadyListener();
      this.updateCellColors();
    });
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

  private buildPrimitive(instances: GeometryInstance[]): Primitive {
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

  private buildColoredCellInstance(cell: GridCell, color: Color): GeometryInstance {
    const instance = this.buildCellInstance(cell);
    instance.attributes = { color: ColorGeometryInstanceAttribute.fromColor(color) };
    return instance;
  }

  private velocityColorForCell(cell: GridCell): Color {
    if (
      !cell.has_velocity ||
      cell.velocity_speed_m_per_s === undefined ||
      !this.dataset?.velocity_p98_m_per_s
    ) {
      return EQUATORIAL_CAUTION_COLOR;
    }
    const normalized = Math.sqrt(Math.min(cell.velocity_speed_m_per_s / this.dataset.velocity_p98_m_per_s, 1));
    return normalized < 0.55
      ? Color.lerp(VELOCITY_LOW_COLOR, VELOCITY_MIDDLE_COLOR, normalized / 0.55, new Color())
      : Color.lerp(VELOCITY_MIDDLE_COLOR, VELOCITY_HIGH_COLOR, (normalized - 0.55) / 0.45, new Color());
  }

  private mldColorForCell(cell: GridCell): Color {
    if (!cell.has_mld || cell.mld_max_m === undefined || !this.dataset?.mld_max_p98_m) {
      return OCEAN_CELL_COLOR;
    }
    const normalized = Math.sqrt(Math.min(cell.mld_max_m / this.dataset.mld_max_p98_m, 1));
    return normalized < 0.5
      ? Color.lerp(MLD_LOW_COLOR, MLD_MIDDLE_COLOR, normalized / 0.5, new Color())
      : Color.lerp(MLD_MIDDLE_COLOR, MLD_HIGH_COLOR, (normalized - 0.5) / 0.5, new Color());
  }

  private subductionColorForCell(cell: GridCell): Color {
    const limit = this.dataset?.subduction_abs_p98_m_per_s;
    if (!cell.has_subduction || cell.subduction_m_per_s === undefined || !limit) {
      return EQUATORIAL_CAUTION_COLOR;
    }
    const normalized = Math.max(-1, Math.min(cell.subduction_m_per_s / limit, 1));
    return normalized < 0
      ? Color.lerp(SUBDUCTION_NEGATIVE_COLOR, SUBDUCTION_ZERO_COLOR, normalized + 1, new Color())
      : Color.lerp(SUBDUCTION_ZERO_COLOR, SUBDUCTION_POSITIVE_COLOR, normalized, new Color());
  }

  private colorForCell(cell: GridCell): Color {
    if (cell.is_land) return LAND_CELL_COLOR;

    if (this.visualizationMode === "velocity") {
      if (!this.showEquatorialEstimates && cell.equatorial_estimate) {
        return EQUATORIAL_CAUTION_COLOR;
      }
      return this.velocityColorForCell(cell);
    }
    if (this.visualizationMode === "mld") return this.mldColorForCell(cell);
    if (this.visualizationMode === "subduction") {
      if (!this.showEquatorialSubduction && cell.equatorial_estimate && cell.has_subduction) {
        return EQUATORIAL_CAUTION_COLOR;
      }
      return this.subductionColorForCell(cell);
    }
    return OCEAN_CELL_COLOR;
  }

  private updateCellColors(): void {
    if (!this.dataset || !this.dataPrimitive?.ready) return;

    for (const cell of this.dataset.cells) {
      const attributes = this.dataPrimitive.getGeometryInstanceAttributes(cell);
      attributes.color = ColorGeometryInstanceAttribute.toValue(
        this.colorForCell(cell),
        attributes.color,
      );
    }
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
