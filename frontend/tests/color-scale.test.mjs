import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

// Use the project's TypeScript compiler without adding a separate test runner.
const source = readFileSync(new URL('../src/globe/ColorScale.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
});
const { quantile, sortedValues, resolveScale, normalize, colorAt, legendStops, defaultRange } =
  await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const scale = (mode, minimum, maximum) => ({ mode, minimum, maximum, lower: 2, upper: 98 });

test('quantiles interpolate ranks and include actual extrema at 0 and 100', () => {
  assert.ok(Math.abs(quantile([0, 10, 20, 30], 40) - 12) < 1e-12);
  assert.equal(quantile([-10, 5, 90], 0), -10);
  assert.equal(quantile([-10, 5, 90], 100), 90);
  assert.equal(quantile([7, 7, 7], 60), 7);
  assert.equal(quantile([4], 98), 4);
});

test('populations exclude land, unavailable and nonfinite values, but retain equatorial estimates', () => {
  const cell = { is_land: 0, has_velocity: 1, velocity_speed_m_per_s: 2 };
  const cells = [cell, { ...cell, velocity_speed_m_per_s: -1, equatorial_estimate: 1 },
    { ...cell, is_land: 1 }, { ...cell, has_velocity: 0 },
    { ...cell, velocity_speed_m_per_s: NaN }, { ...cell, velocity_speed_m_per_s: Infinity },
    { ...cell, velocity_speed_m_per_s: undefined }];
  assert.deepEqual(sortedValues(cells, 'velocity'), [-1, 2]);
  assert.deepEqual(sortedValues([{ is_land: 0, has_mld: 1, mld_max_m: 40 }], 'mld'), [40]);
  assert.deepEqual(sortedValues([{ is_land: 0, has_subduction: 1, subduction_m_per_s: -3 }], 'subduction'), [-3]);
});

test('default ranges are independent and empty populations have no scale', () => {
  const first = defaultRange();
  first.lower = 40;
  assert.deepEqual(defaultRange(), { lower: 2, upper: 98 });
  assert.equal(resolveScale('mld', [], defaultRange()), null);
  assert.equal(quantile([], 50), undefined);
  assert.deepEqual(resolveScale('mld', [0, 100], { lower: 40, upper: 60 }),
    { mode: 'mld', lower: 40, upper: 60, minimum: 40, maximum: 60 });
});

test('sequential scales stretch the chosen interval with square-root normalization and clip', () => {
  for (const mode of ['velocity', 'mld']) {
    const s = scale(mode, 40, 60);
    assert.equal(normalize(0, s), 0);
    assert.equal(normalize(45, s), 0.5);
    assert.equal(normalize(100, s), 1);
    assert.deepEqual(colorAt(-20, s), colorAt(40, s));
    assert.deepEqual(colorAt(120, s), colorAt(60, s));
  }
});

test('asymmetric signed scale keeps neutral at zero and clips each endpoint', () => {
  const s = scale('subduction', -2, 8);
  assert.equal(normalize(-2, s), 0);
  assert.equal(normalize(-1, s), 0.25);
  assert.equal(normalize(0, s), 0.5);
  assert.equal(normalize(4, s), 0.75);
  assert.equal(normalize(8, s), 1);
  assert.deepEqual(colorAt(0, s), [255, 255, 191]);
  assert.deepEqual(colorAt(-20, s), colorAt(-2, s));
  assert.deepEqual(colorAt(20, s), colorAt(8, s));
  assert.match(legendStops(s), /rgb\(255,255,191\) 20%/);
});

test('one-sign ranges use only the corresponding side of the signed palette', () => {
  const positive = scale('subduction', 2, 8);
  const negative = scale('subduction', -8, -2);
  assert.equal(normalize(2, positive), 0.625);
  assert.equal(normalize(8, positive), 1);
  assert.equal(normalize(-8, negative), 0);
  assert.equal(normalize(-2, negative), 0.375);
});

test('constant scales produce one finite color including signed zero', () => {
  for (const mode of ['mld', 'velocity', 'subduction']) {
    for (const value of [-3, 0, 3]) {
      const s = scale(mode, value, value);
      assert.ok(Number.isFinite(normalize(value, s)));
      assert.deepEqual(colorAt(-100, s), colorAt(100, s));
      assert.ok(!/NaN|Infinity/.test(legendStops(s)));
    }
  }
});
