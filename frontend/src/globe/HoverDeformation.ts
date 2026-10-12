import {
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
} from "cesium";

import type { GridCell, GridDataset } from "./GridLayer";

const MAX_RISE_METERS = 3200.0;
const HOVER_SIGMA_CELLS = 1.35;
const NEIGHBOR_RADIUS = 3;
const EASE_SPEED = 7.0;

type WeightedCell = {
  cell: GridCell;
  weight: number;
};

export class HoverDeformation {
  private scene: Scene;
  private dataset: GridDataset;
  private hoveredCellId: number | null = null;
  private targetStrength = 0;
  private currentStrength = 0;
  private neighborhood: WeightedCell[] = [];
  private primitive: Primitive | null = null;
  private destroyed = false;
  private lastRenderedStrength = -1;
  private lastRenderedHoverId: number | null = null;
  private readonly postRenderHandler: () => void;

  constructor(viewer: Viewer, dataset: GridDataset) {
    this.scene = viewer.scene;
    this.dataset = dataset;
    this.postRenderHandler = () => this.tick();
    this.scene.postRender.addEventListener(this.postRenderHandler);
  }

  updateHoveredCell(cell: GridCell | null): void {
    if (!cell || cell.is_land) {
      this.targetStrength = 0;
      this.hoveredCellId = null;
      return;
    }

    if (this.hoveredCellId === cell.id) {
      this.targetStrength = 1;
      return;
    }

    this.hoveredCellId = cell.id;
    this.targetStrength = 1;
    this.neighborhood = this.collectNeighborhood(cell);
  }

  destroy(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    this.scene.postRender.removeEventListener(this.postRenderHandler);
    this.clearPrimitive();
  }

  private tick(): void {
    if (this.destroyed) {
      return;
    }

    const delta = this.targetStrength - this.currentStrength;
    if (Math.abs(delta) < 0.002) {
      this.currentStrength = this.targetStrength;
    } else {
      this.currentStrength += delta * 0.12 * EASE_SPEED;
      this.currentStrength = Math.max(0, Math.min(1, this.currentStrength));
    }

    if (this.currentStrength <= 0.001) {
      this.lastRenderedStrength = -1;
      this.lastRenderedHoverId = null;
      this.clearPrimitive();
      return;
    }

    if (
      Math.abs(this.currentStrength - this.lastRenderedStrength) < 0.01 &&
      this.hoveredCellId === this.lastRenderedHoverId
    ) {
      return;
    }

    this.rebuildPrimitive();
    this.lastRenderedStrength = this.currentStrength;
    this.lastRenderedHoverId = this.hoveredCellId;
  }

  private rebuildPrimitive(): void {
    this.clearPrimitive();
    const instances: GeometryInstance[] = [];

    for (const item of this.neighborhood) {
      const isHovered = this.hoveredCellId === item.cell.id;
      const rise = MAX_RISE_METERS * item.weight * this.currentStrength;
      const alpha = isHovered ? 0.98 : 0.82;
      const color = isHovered ? "#66b3ff" : "#4f90dd";
      instances.push(
        new GeometryInstance({
          geometry: this.buildCellGeometry(item.cell, rise + 7.0),
          id: item.cell,
          attributes: {
            color: ColorGeometryInstanceAttribute.fromColor(
              Color.fromCssColorString(color).withAlpha(alpha),
            ),
          },
        }),
      );
    }

    this.primitive = new Primitive({
      geometryInstances: instances,
      appearance: new PerInstanceColorAppearance({
        translucent: true,
        closed: false,
        flat: true,
        renderState: {
          depthTest: { enabled: true },
          depthMask: true,
        },
      }),
      asynchronous: false,
      releaseGeometryInstances: true,
    });
    this.scene.primitives.add(this.primitive);
  }

  private buildCellGeometry(cell: GridCell, height: number): PolygonGeometry {
    const lat0 = cell.lat0 <= -90 ? -89.999 : cell.lat0;
    const lat1 = cell.lat1 >= 90 ? 89.999 : cell.lat1;
    return new PolygonGeometry({
      polygonHierarchy: new PolygonHierarchy(
        Cartesian3.fromDegreesArray([
          cell.lon0,
          lat0,
          cell.lon1,
          lat0,
          cell.lon1,
          lat1,
          cell.lon0,
          lat1,
        ]),
      ),
      height,
      vertexFormat: PerInstanceColorAppearance.FLAT_VERTEX_FORMAT,
    });
  }

  private collectNeighborhood(center: GridCell): WeightedCell[] {
    const out: WeightedCell[] = [];
    const { n_lat, n_lon, cells } = this.dataset;
    const sigma2 = HOVER_SIGMA_CELLS * HOVER_SIGMA_CELLS;

    for (let di = -NEIGHBOR_RADIUS; di <= NEIGHBOR_RADIUS; di += 1) {
      const i = center.i + di;
      if (i < 0 || i >= n_lat) {
        continue;
      }

      for (let dj = -NEIGHBOR_RADIUS; dj <= NEIGHBOR_RADIUS; dj += 1) {
        const wrappedJ = ((center.j + dj) % n_lon + n_lon) % n_lon;
        const cellId = i * n_lon + wrappedJ;
        const cell = cells[cellId];
        if (!cell || cell.is_land) {
          continue;
        }

        const wrappedDj = Math.min(Math.abs(dj), n_lon - Math.abs(dj));
        const d2 = di * di + wrappedDj * wrappedDj;
        const weight = Math.exp(-(d2 / (2.0 * sigma2)));
        if (weight < 0.03) {
          continue;
        }
        out.push({ cell, weight });
      }
    }

    return out;
  }

  private clearPrimitive(): void {
    if (this.primitive) {
      this.scene.primitives.remove(this.primitive);
      this.primitive.destroy();
      this.primitive = null;
    }
  }
}
