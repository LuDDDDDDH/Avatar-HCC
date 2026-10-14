import fs from 'node:fs/promises';
import ExcelJS from 'exceljs';
import { parse } from 'csv-parse/sync';
import sharp from 'sharp';
import { normalizeSpreadsheetNamespaces, preserveNativeCharts, restoreConditionalRules, preserveConditionalRules } from './xlsx_parts.mjs';

const clone = value => value === undefined ? undefined : structuredClone(value);
const xml = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&apos;');
const colour = value => {
  const result = typeof value === 'string' ? value.replace(/^#/, '').toUpperCase() : value?.argb;
  return result?.length === 6 ? `FF${result}` : result;
};
const cssColour = (value, fallback = '#FFFFFF') => {
  const argb = colour(value);
  return argb ? `#${argb.length === 8 ? argb.slice(2) : argb}` : fallback;
};

function colName(index) {
  if (!Number.isSafeInteger(index) || index < 1 || index > 16384) throw new Error('Invalid column index');
  let result = '';
  for (let n = index; n > 0; n = Math.floor((n - 1) / 26)) result = String.fromCharCode(65 + (n - 1) % 26) + result;
  return result;
}

function parseAddress(address) {
  if (typeof address !== 'string') throw new Error('Range must be an A1 address');
  const match = /^\$?([A-Za-z]{1,3})\$?(\d+)(?::\$?([A-Za-z]{1,3})\$?(\d+))?$/.exec(address);
  if (!match) throw new Error(`Unsupported range: ${address}`);
  const toCol = name => [...name.toUpperCase()].reduce((n, letter) => n * 26 + letter.charCodeAt(0) - 64, 0);
  const c1 = toCol(match[1]), r1 = Number(match[2]);
  const c2 = match[3] ? toCol(match[3]) : c1, r2 = match[4] ? Number(match[4]) : r1;
  if (r1 < 1 || r2 < r1 || c1 < 1 || c2 < c1 || c2 > 16384 || r2 > 1048576) throw new Error(`Invalid range: ${address}`);
  return { r1, c1, r2, c2 };
}

function mergeBounds(sheet) {
  return (sheet.model.merges ?? []).map(address => {
    const item = parseAddress(address);
    return { top: item.r1, left: item.c1, bottom: item.r2, right: item.c2 };
  });
}

function cellValue(cell) {
  if (cell.type === ExcelJS.ValueType.Merge) return null;
  const value = cell.value;
  if (value === undefined || value === null) return null;
  if (typeof value !== 'object' || value instanceof Date) return value;
  if (Object.hasOwn(value, 'formula') || Object.hasOwn(value, 'sharedFormula')) {
    const result = value.result;
    return result && typeof result === 'object' && result.error ? result.error : result ?? null;
  }
  if (Object.hasOwn(value, 'error')) return value.error;
  if (Array.isArray(value.richText)) return value.richText.map(part => part.text).join('');
  if (Object.hasOwn(value, 'text')) return value.text;
  throw new Error(`Unsupported cell value at ${cell.address}`);
}

function setCellValue(cell, value) {
  if (cell.type === ExcelJS.ValueType.Merge) {
    if (value !== null && value !== undefined && value !== '') throw new Error(`Cannot write a merged-cell follower: ${cell.address}`);
    return;
  }
  if (value !== null && value !== undefined && !(value instanceof Date) && !['string', 'boolean', 'number'].includes(typeof value)) throw new Error(`Unsupported cell value at ${cell.address}`);
  if (typeof value === 'number' && !Number.isFinite(value)) throw new Error(`Non-finite cell value at ${cell.address}`);
  cell.value = value === undefined ? null : value;
}

function fontPart(font, key, value) {
  const result = { ...clone(font) };
  if (!['name', 'size', 'bold', 'italic', 'underline', 'color', 'strike', 'family', 'scheme'].includes(key)) throw new Error(`Unsupported font property: ${key}`);
  result[key] = key === 'color' ? { argb: colour(value) } : value;
  return result;
}

function mergedFont(font, changes) {
  let result = clone(font) ?? {};
  for (const [key, value] of Object.entries(changes)) result = fontPart(result, key, value);
  return result;
}

function mergedBorder(border, changes) {
  const result = clone(border) ?? {};
  for (const [side, part] of Object.entries(changes)) {
    if (!['top', 'right', 'bottom', 'left', 'diagonal'].includes(side)) throw new Error(`Unsupported border side: ${side}`);
    const next = { ...(result[side] ?? {}) };
    for (const [key, value] of Object.entries(part)) {
      if (!['style', 'color', 'up', 'down'].includes(key)) throw new Error(`Unsupported border property: ${key}`);
      next[key] = key === 'color' ? { argb: colour(value) } : value;
    }
    result[side] = next;
  }
  return result;
}

class Range {
  constructor(sheet, address) {
    this.sheet = sheet;
    Object.assign(this, parseAddress(address));
    this._format = createFormat(this);
  }
  get rowCount() { return this.r2 - this.r1 + 1; }
  get columnCount() { return this.c2 - this.c1 + 1; }
  get address() { return `${this.sheet.name}!${colName(this.c1)}${this.r1}:${colName(this.c2)}${this.r2}`; }
  each(callback) {
    for (let r = this.r1; r <= this.r2; r++) for (let c = this.c1; c <= this.c2; c++) callback(this.sheet._sheet.getCell(r, c), r, c);
  }
  styleEach(callback) {
    this.each(cell => { cell.style = clone(cell.style) ?? {}; callback(cell); });
  }
  get values() {
    return Array.from({ length: this.rowCount }, (_, i) => Array.from({ length: this.columnCount }, (_, j) => cellValue(this.sheet._sheet.getCell(this.r1 + i, this.c1 + j))));
  }
  set values(matrix) {
    if (!Array.isArray(matrix) || matrix.length > this.rowCount || matrix.some(row => !Array.isArray(row) || row.length > this.columnCount)) throw new Error(`Value matrix exceeds range ${this.address}`);
    matrix.forEach((row, i) => row.forEach((value, j) => setCellValue(this.sheet._sheet.getCell(this.r1 + i, this.c1 + j), value)));
  }
  get formulas() {
    return Array.from({ length: this.rowCount }, (_, i) => Array.from({ length: this.columnCount }, (_, j) => {
      const cell = this.sheet._sheet.getCell(this.r1 + i, this.c1 + j);
      return cell.type !== ExcelJS.ValueType.Merge && cell.formula ? `=${cell.formula}` : null;
    }));
  }
  get format() { return this._format; }
  set format(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Format must be an object');
    for (const [key, item] of Object.entries(value)) this._format[key] = item;
  }
  setNumberFormat(value) { this.format.numberFormat = value; }
  clear({ applyTo = 'all' } = {}) {
    if (!['all', 'contents', 'formats'].includes(applyTo)) throw new Error(`Unsupported clear mode: ${applyTo}`);
    this.each(cell => {
      if (applyTo !== 'formats' && cell.type !== ExcelJS.ValueType.Merge) cell.value = null;
      if (applyTo !== 'contents') cell.style = {};
    });
  }
  copyFrom(source, mode = 'all') {
    if (!(source instanceof Range) || mode !== 'all') throw new Error('copyFrom supports an explicit range and all mode');
    if (this.rowCount !== source.rowCount || this.columnCount !== source.columnCount) throw new Error('copyFrom range dimensions differ');
    const items = [];
    source.each(cell => {
      if (cell.formula) throw new Error('Relative formula copying is unsupported');
      items.push({ value: clone(cell.value), style: clone(cell.style), type: cell.type });
    });
    let index = 0;
    this.each(cell => {
      const item = items[index++];
      if (item.type !== ExcelJS.ValueType.Merge && cell.type !== ExcelJS.ValueType.Merge) cell.value = item.value;
      cell.style = item.style;
    });
  }
  get conditionalFormats() {
    return { deleteAll: () => {
      const intersects = address => {
        const other = parseAddress(address);
        return !(other.r2 < this.r1 || other.r1 > this.r2 || other.c2 < this.c1 || other.c1 > this.c2);
      };
      this.sheet._sheet.removeConditionalFormatting(rule => {
        const references = rule.ref.split(/\s+/);
        const retained = references.flatMap(reference => {
          if (!intersects(reference)) return [reference];
          const other = parseAddress(reference);
          const pieces = [];
          const add = (r1, c1, r2, c2) => {
            if (r1 <= r2 && c1 <= c2) pieces.push(`${colName(c1)}${r1}:${colName(c2)}${r2}`);
          };
          add(other.r1, other.c1, Math.min(other.r2, this.r1 - 1), other.c2);
          add(Math.max(other.r1, this.r2 + 1), other.c1, other.r2, other.c2);
          const r1 = Math.max(other.r1, this.r1), r2 = Math.min(other.r2, this.r2);
          add(r1, other.c1, r2, Math.min(other.c2, this.c1 - 1));
          add(r1, Math.max(other.c1, this.c2 + 1), r2, other.c2);
          return pieces;
        });
        if (retained.length) rule.ref = retained.join(' ');
        return retained.length > 0;
      });
    } };
  }
}

function createFormat(range) {
  const first = () => range.sheet._sheet.getCell(range.r1, range.c1);
  const nested = (group, side) => new Proxy({}, {
    get(_target, key) {
      if (group === 'font') return key === 'color' ? cssColour(first().font?.color, '#000000') : first().font?.[key];
      if (!side) return nested('border', key);
      return key === 'color' ? cssColour(first().border?.[side]?.color, '#000000') : first().border?.[side]?.[key];
    },
    set(_target, key, value) {
      range.styleEach(cell => {
        if (group === 'font') cell.font = fontPart(cell.font, key, value);
        else cell.border = mergedBorder(cell.border, side ? { [side]: { [key]: value } } : { [key]: value });
      });
      return true;
    },
  });
  return new Proxy({}, {
    get(_target, key) {
      if (key === 'font') return nested('font');
      if (key === 'borders') return nested('border');
      if (key === 'autofitRows') return () => autoFitRows(range);
      if (key === 'fill') return cssColour(first().fill?.fgColor);
      if (key === 'numberFormat') return first().numFmt;
      if (['horizontalAlignment', 'verticalAlignment', 'wrapText'].includes(key)) return first().alignment?.[{ horizontalAlignment: 'horizontal', verticalAlignment: 'vertical', wrapText: 'wrapText' }[key]];
      if (key === 'rowHeight') return range.sheet._sheet.getRow(range.r1).height;
      if (key === 'rowHeightPx') return (range.sheet._sheet.getRow(range.r1).height ?? 15) * 4 / 3;
      if (key === 'columnWidthPx') return colPixels(range.sheet._sheet, range.c1);
      throw new Error(`Unsupported format property: ${String(key)}`);
    },
    set(_target, key, value) {
      if (key === 'rowHeight' || key === 'rowHeightPx') {
        if (!Number.isFinite(value) || value < 0) throw new Error('Invalid row height');
        for (let r = range.r1; r <= range.r2; r++) range.sheet._sheet.getRow(r).height = key === 'rowHeight' ? value : value * 3 / 4;
      } else if (key === 'columnWidthPx') {
        if (!Number.isFinite(value) || value < 0) throw new Error('Invalid column width');
        for (let c = range.c1; c <= range.c2; c++) range.sheet._sheet.getColumn(c).width = Math.fround(Number((value / 8).toFixed(2)));
      } else if (key === 'font') range.styleEach(cell => { cell.font = mergedFont(cell.font, value); });
      else if (key === 'borders') range.styleEach(cell => { cell.border = mergedBorder(cell.border, value); });
      else if (key === 'fill') range.styleEach(cell => { cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: colour(value) } }; });
      else if (key === 'numberFormat') range.styleEach(cell => { cell.numFmt = value; });
      else if (['horizontalAlignment', 'verticalAlignment', 'wrapText'].includes(key)) {
        const mapped = { horizontalAlignment: 'horizontal', verticalAlignment: 'vertical', wrapText: 'wrapText' }[key];
        range.styleEach(cell => { cell.alignment = { ...clone(cell.alignment), [mapped]: key === 'verticalAlignment' && value === 'center' ? 'middle' : value }; });
      } else throw new Error(`Unsupported format property: ${String(key)}`);
      return true;
    },
  });
}

function colPixels(sheet, index) {
  const width = sheet.getColumn(index).width ?? sheet.properties.defaultColWidth ?? 9.14;
  return width * 8;
}

function rowPixels(sheet, index) {
  const row = sheet.getRow(index);
  return row.hidden ? 0 : (row.height ?? sheet.properties.defaultRowHeight ?? 15) * 4 / 3;
}

function textWidth(value, fontSize) {
  return [...value].reduce((sum, char) => sum + (/\s/.test(char) ? 0.3 : /[ilI1.,:;'|!]/.test(char) ? 0.28 : /[MW@#%]/.test(char) ? 0.85 : char.codePointAt(0) > 255 ? 0.85 : 0.56) * fontSize, 0);
}

function wrappedLines(value, width, fontSize, wrap) {
  const lines = [];
  for (const line of String(value ?? '').split('\n')) {
    if (!wrap || textWidth(line, fontSize) <= width) { lines.push(line); continue; }
    let current = '';
    for (const token of line.split(/(\s+)/)) {
      if (current && textWidth(current + token, fontSize) > width) { lines.push(current.trimEnd()); current = token.trimStart(); }
      else current += token;
      while (textWidth(current, fontSize) > width && current.length > 1) {
        let end = 1;
        while (end < current.length && textWidth(current.slice(0, end + 1), fontSize) <= width) end++;
        lines.push(current.slice(0, end)); current = current.slice(end);
      }
    }
    lines.push(current.trimEnd());
  }
  return lines;
}

function autoFitRows(range) {
  const sheet = range.sheet._sheet;
  for (let r = range.r1; r <= range.r2; r++) {
    let required = 0;
    for (let c = range.c1; c <= range.c2; c++) {
      const cell = sheet.getCell(r, c);
      if (cell.type === ExcelJS.ValueType.Merge) continue;
      const size = (cell.font?.size ?? 11) * 4 / 3;
      let width = colPixels(sheet, c);
      const merge = mergeBounds(sheet).find(item => item.top === r && item.left === c);
      if (merge) {
        if (merge.bottom !== merge.top) throw new Error('Automatic row height for a multi-row merge is unsupported');
        width = 0; for (let mc = merge.left; mc <= merge.right; mc++) width += colPixels(sheet, mc);
      }
      const text = displayedValue(cell);
      const count = wrappedLines(text, Math.max(1, width - 12), size, Boolean(cell.alignment?.wrapText)).length;
      required = Math.max(required, count * size * 1.25 + 10);
    }
    sheet.getRow(r).height = Math.max(sheet.getRow(r).height ?? 15, required * 3 / 4);
  }
}

class Sheet {
  constructor(workbook, sheet) { this.workbook = workbook; this._sheet = sheet; }
  get name() { return this._sheet.name; }
  get showGridLines() { return this._sheet.views?.[0]?.showGridLines ?? true; }
  set showGridLines(value) { this._sheet.views = [{ ...(this._sheet.views?.[0] ?? {}), showGridLines: Boolean(value) }]; }
  get tabColor() { return cssColour(this._sheet.properties.tabColor); }
  set tabColor(value) { this._sheet.properties.tabColor = { argb: colour(value) }; }
  get freezePanes() {
    const freeze = (key, count) => {
      if (!Number.isSafeInteger(count) || count < 0) throw new Error('Invalid freeze count');
      const view = { ...(this._sheet.views?.[0] ?? {}), state: 'frozen', [key]: count };
      view.topLeftCell = `${colName((view.xSplit ?? 0) + 1)}${(view.ySplit ?? 0) + 1}`;
      this._sheet.views = [view];
    };
    return { freezeRows: count => freeze('ySplit', count), freezeColumns: count => freeze('xSplit', count) };
  }
  getRange(address) { return new Range(this, address); }
  getRangeByIndexes(row, column, rows, columns) {
    if (![row, column, rows, columns].every(Number.isSafeInteger) || row < 0 || column < 0 || rows < 1 || columns < 1) throw new Error('Invalid indexed range');
    return this.getRange(`${colName(column + 1)}${row + 1}:${colName(column + columns)}${row + rows}`);
  }
  getUsedRange(valuesOnly = false) {
    let r1 = Infinity, c1 = Infinity, r2 = 0, c2 = 0;
    this._sheet.eachRow({ includeEmpty: true }, row => row.eachCell({ includeEmpty: true }, cell => {
      const isValue = cellValue(cell) !== null || Boolean(cell.formula);
      const isStyled = !valuesOnly && Object.keys(cell.style ?? {}).length > 0;
      if (!isValue && !isStyled) return;
      r1 = Math.min(r1, cell.row); c1 = Math.min(c1, cell.col); r2 = Math.max(r2, cell.row); c2 = Math.max(c2, cell.col);
    }));
    if (!r2) return null;
    if (!valuesOnly) for (const item of mergeBounds(this._sheet)) {
      r1 = Math.min(r1, item.top); c1 = Math.min(c1, item.left); r2 = Math.max(r2, item.bottom); c2 = Math.max(c2, item.right);
    }
    return this.getRange(`${colName(c1)}${r1}:${colName(c2)}${r2}`);
  }
  mergeCells(address) { parseAddress(address); this._sheet.mergeCells(address); }
}

function displayedValue(cell) {
  const value = cellValue(cell);
  if (value === null) return '';
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  if (typeof value !== 'number') return String(value);
  const format = cell.numFmt;
  if (!format || format === 'General') return String(value);
  if (/^0(?:\.(0+))?E\+00$/i.test(format)) {
    const decimals = (format.match(/\.(0+)/)?.[1] ?? '').length;
    const [mantissa, exponent] = value.toExponential(decimals).split('e');
    const number = Number(exponent);
    return `${mantissa}E${number >= 0 ? '+' : '-'}${String(Math.abs(number)).padStart(2, '0')}`;
  }
  if (/^0(?:\.0+)?$/.test(format)) return value.toFixed((format.split('.')[1] ?? '').length);
  if (/^#,##0(?:\.0+)?$/.test(format)) return value.toLocaleString('en-US', { minimumFractionDigits: (format.split('.')[1] ?? '').length, maximumFractionDigits: (format.split('.')[1] ?? '').length });
  if (/^0(?:\.0+)?%$/.test(format)) return `${(value * 100).toFixed((format.match(/\.([0]+)%/)?.[1] ?? '').length)}%`;
  if (format === '@') return String(value);
  throw new Error(`Unsupported numeric display format: ${format}`);
}

function renderSvg(range, fitRows = false) {
  const sheet = range.sheet._sheet;
  const xs = [0], ys = [0];
  const merges = mergeBounds(sheet);
  for (let c = range.c1; c <= range.c2; c++) xs.push(xs.at(-1) + (sheet.getColumn(c).hidden ? 0 : colPixels(sheet, c)));
  for (let r = range.r1; r <= range.r2; r++) {
    let height = rowPixels(sheet, r);
    if (fitRows && height > 0 && !merges.some(item => item.top !== item.bottom && r >= item.top && r <= item.bottom)) {
      for (let c = range.c1; c <= range.c2; c++) {
        const merge = merges.find(item => r >= item.top && r <= item.bottom && c >= item.left && c <= item.right);
        if (merge && c !== Math.max(merge.left, range.c1)) continue;
        const cell = sheet.getCell(r, merge ? merge.left : c);
        const left = c - range.c1;
        const right = merge ? Math.min(merge.right, range.c2) - range.c1 + 1 : left + 1;
        const width = xs[right] - xs[left];
        if (width <= 0) continue;
        const size = (cell.font?.size ?? 11) * 4 / 3;
        const lines = wrappedLines(displayedValue(cell), Math.max(1, width - 12), size, Boolean(cell.alignment?.wrapText));
        height = Math.max(height, lines.length * size * 1.25 + 12);
      }
    }
    ys.push(ys.at(-1) + height);
  }
  const width = Math.ceil(xs.at(-1)) + 1, height = Math.ceil(ys.at(-1)) + 1;
  const elements = [`<rect x="0" y="0" width="${width}" height="${height}" fill="#FFFFFF"/>`];
  const definitions = [];
  let clipIndex = 0;
  for (let r = range.r1; r <= range.r2; r++) for (let c = range.c1; c <= range.c2; c++) {
    const cell = sheet.getCell(r, c);
    const merge = merges.find(item => r >= item.top && r <= item.bottom && c >= item.left && c <= item.right);
    if (merge && (r !== Math.max(merge.top, range.r1) || c !== Math.max(merge.left, range.c1))) continue;
    const anchor = merge ? sheet.getCell(merge.top, merge.left) : cell;
    const i = c - range.c1, j = r - range.r1;
    const x = xs[i], y = ys[j];
    const right = merge ? Math.min(merge.right, range.c2) - range.c1 + 1 : i + 1;
    const bottom = merge ? Math.min(merge.bottom, range.r2) - range.r1 + 1 : j + 1;
    const w = xs[right] - x, h = ys[bottom] - y;
    if (w <= 0 || h <= 0) continue;
    elements.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${cssColour(anchor.fill?.fgColor)}"/>`);
    const border = anchor.border ?? {};
    const sides = { top: [x, y, x + w, y], right: [x + w, y, x + w, y + h], bottom: [x, y + h, x + w, y + h], left: [x, y, x, y + h] };
    for (const [side, line] of Object.entries(sides)) {
      const edge = border[side];
      if (edge?.style && edge.style !== 'none') {
        const weight = /thick|double/.test(edge.style) ? 3 : /medium/.test(edge.style) ? 2 : 1;
        const dash = /dash|dotted/i.test(edge.style) ? ' stroke-dasharray="4 3"' : '';
        elements.push(`<line x1="${line[0]}" y1="${line[1]}" x2="${line[2]}" y2="${line[3]}" stroke="${cssColour(edge.color, '#AAB6C2')}" stroke-width="${weight}"${dash}/>`);
      } else if (range.sheet.showGridLines) elements.push(`<line x1="${line[0]}" y1="${line[1]}" x2="${line[2]}" y2="${line[3]}" stroke="#D9DEE5" stroke-width="0.6"/>`);
    }
    const text = displayedValue(anchor);
    const size = (anchor.font?.size ?? 11) * 4 / 3;
    const lines = wrappedLines(text, Math.max(1, w - 12), size, Boolean(anchor.alignment?.wrapText));
    const lineHeight = size * 1.25;
    const vertical = anchor.alignment?.vertical ?? 'bottom';
    const initialY = vertical === 'top' ? y + 6 + size : vertical === 'center' || vertical === 'middle' ? y + Math.max(size, (h - lines.length * lineHeight) / 2 + size) : y + h - 6 - (lines.length - 1) * lineHeight;
    const align = anchor.alignment?.horizontal ?? (typeof cellValue(anchor) === 'number' ? 'right' : 'left');
    if (!['left', 'center', 'right', 'general'].includes(align)) throw new Error(`Unsupported text alignment: ${align}`);
    const tx = align === 'center' ? x + w / 2 : align === 'right' ? x + w - 6 : x + 6;
    const textAnchor = align === 'center' ? 'middle' : align === 'right' ? 'end' : 'start';
    const id = `cell${clipIndex++}`;
    definitions.push(`<clipPath id="${id}"><rect x="${x + 1}" y="${y + 1}" width="${Math.max(0, w - 2)}" height="${Math.max(0, h - 2)}"/></clipPath>`);
    elements.push(`<g clip-path="url(#${id})"><text x="${tx}" y="${initialY}" font-family="${xml(anchor.font?.name ?? 'Arial')}" font-size="${size}" font-weight="${anchor.font?.bold ? 'bold' : 'normal'}" font-style="${anchor.font?.italic ? 'italic' : 'normal'}" fill="${cssColour(anchor.font?.color, '#1F2937')}" text-anchor="${textAnchor}">${lines.map((line, index) => `<tspan x="${tx}" dy="${index ? lineHeight : 0}">${xml(line)}</tspan>`).join('')}</text></g>`);
  }
  return { svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><defs>${definitions.join('')}</defs>${elements.join('')}</svg>`, width, height };
}

export class FileBlob {
  constructor(buffer) { this.buffer = Buffer.from(buffer); }
  static async load(path) { return new FileBlob(await fs.readFile(path)); }
  async save(path) { await fs.writeFile(path, this.buffer); }
  async arrayBuffer() { return this.buffer.buffer.slice(this.buffer.byteOffset, this.buffer.byteOffset + this.buffer.byteLength); }
}

export class Workbook {
  constructor(workbook = new ExcelJS.Workbook()) { this._workbook = workbook; }
  static create() { return new Workbook(); }
  static async fromCSV(text, { sheetName = 'Data' } = {}) {
    const rows = parse(text, { bom: false, skip_empty_lines: false, relax_column_count: true, cast: false });
    const workbook = Workbook.create();
    const sheet = workbook.worksheets.add(sheetName);
    if (rows.length) {
      const columns = Math.max(...rows.map(row => row.length));
      sheet.getRangeByIndexes(0, 0, rows.length, columns).values = rows.map(row => Array.from({ length: columns }, (_, i) => row[i] ?? ''));
    }
    return workbook;
  }
  get worksheets() {
    return {
      get items() { return this.owner._workbook.worksheets.map(sheet => new Sheet(this.owner, sheet)); },
      owner: this,
      add: name => new Sheet(this, this._workbook.addWorksheet(name)),
      getItem: name => {
        const sheet = this._workbook.getWorksheet(name);
        if (!sheet) throw new Error(`Unknown worksheet: ${name}`);
        return new Sheet(this, sheet);
      },
      getItemAt: index => {
        const sheet = this._workbook.worksheets[index];
        if (!sheet) throw new Error(`Unknown worksheet index: ${index}`);
        return new Sheet(this, sheet);
      },
    };
  }
  recalculate() {
    for (const sheet of this._workbook.worksheets) sheet.eachRow(row => row.eachCell(cell => {
      if (cell.formula) throw new Error(`Formula calculation is unsupported: ${sheet.name}!${cell.address}`);
    }));
  }
  async inspect(options = {}) {
    const { kind, range, searchTerm = '', include = '', options: matchOptions = {}, tableMaxRows = 20, tableMaxCols = 20, tableMaxCellChars = 500, maxChars = Infinity } = options;
    let entries;
    if (kind === 'sheet') entries = this._workbook.worksheets.map(sheet => ({ kind: 'sheet', id: String(sheet.id), name: sheet.name }));
    else if (kind === 'table') {
      const index = range?.lastIndexOf('!') ?? -1;
      if (index < 1) throw new Error('Table inspection requires a worksheet range');
      let name = range.slice(0, index);
      if (name.startsWith("'") && name.endsWith("'")) name = name.slice(1, -1).replaceAll("''", "'");
      const selected = this.worksheets.getItem(name).getRange(range.slice(index + 1));
      const cut = matrix => matrix.slice(0, tableMaxRows).map(row => row.slice(0, tableMaxCols).map(value => typeof value === 'string' ? value.slice(0, tableMaxCellChars) : value));
      const item = { kind: 'table', range: selected.address, rowCount: selected.rowCount, columnCount: selected.columnCount };
      if (!include || include.includes('values')) item.values = cut(selected.values);
      if (include.includes('formulas')) item.formulas = cut(selected.formulas);
      entries = [item];
    } else if (kind === 'match') {
      const regex = matchOptions.useRegex ? new RegExp(searchTerm) : null;
      const matches = [], maximum = matchOptions.maxResults ?? 100;
      let count = 0;
      for (const sheet of this.worksheets.items) {
        const used = sheet.getUsedRange(true);
        if (!used) continue;
        used.each(cell => {
          const value = cellValue(cell);
          if (value === null) return;
          const text = String(value);
          const matchesCell = regex ? regex.test(text) : text.includes(searchTerm);
          if (matchesCell) {
            count++;
            if (matches.length < maximum) matches.push({ sheet: sheet.name, cell: cell.address, value });
          }
        });
      }
      entries = [{ kind: 'match', summary: `matched ${count} entries`, matchCount: count, truncated: count > matches.length, matches }];
    } else throw new Error(`Unsupported inspection kind: ${kind}`);
    let output = entries.map(entry => JSON.stringify(entry)).join('\n');
    if (kind !== 'sheet') {
      while (output.length > maxChars) {
        const entry = entries[0];
        if (kind === 'table' && ((entry.values?.length ?? 0) > 0 || (entry.formulas?.length ?? 0) > 0)) {
          entry.values?.pop(); entry.formulas?.pop(); entry.truncated = true;
        } else if (kind === 'match' && entry.matches.length) {
          entry.matches.pop(); entry.truncated = true;
        } else throw new Error('Inspection output exceeds maxChars');
        output = entries.map(item => JSON.stringify(item)).join('\n');
      }
    }
    if (output.length > maxChars) throw new Error('Inspection output exceeds maxChars');
    return { ndjson: output };
  }
  async render({ sheetName, range, scale = 1, format = 'png', fitRows = false } = {}) {
    if (format !== 'png' || !Number.isFinite(scale) || scale <= 0 || scale > 8) throw new Error('Unsupported render format or scale');
    if (typeof fitRows !== 'boolean') throw new Error('fitRows must be a boolean');
    const original = this.worksheets.getItem(sheetName);
    const preview = Workbook.create();
    const sheet = preview.worksheets.add(original.name);
    const model = clone(original._sheet.model);
    model.mergeCells = model.merges;
    sheet._sheet.model = model;
    const rendered = renderSvg(sheet.getRange(range), fitRows);
    const buffer = await sharp(Buffer.from(rendered.svg), { limitInputPixels: 100000000 }).resize(Math.ceil(rendered.width * scale), Math.ceil(rendered.height * scale)).png().toBuffer();
    return new FileBlob(buffer);
  }
}

export class SpreadsheetFile {
  static async importXlsx(blob) {
    if (!(blob instanceof FileBlob)) throw new Error('importXlsx requires a FileBlob');
    const workbook = new ExcelJS.Workbook();
    await workbook.xlsx.load(await normalizeSpreadsheetNamespaces(blob.buffer));
    await restoreConditionalRules(blob.buffer, workbook);
    const result = new Workbook(workbook);
    result._sourceXlsx = blob.buffer;
    return result;
  }
  static async exportXlsx(workbook) {
    if (!(workbook instanceof Workbook)) throw new Error('exportXlsx requires a Workbook');
    const output = await workbook._workbook.xlsx.writeBuffer();
    const conditional = await preserveConditionalRules(output, workbook._workbook);
    return new FileBlob(await preserveNativeCharts(workbook._sourceXlsx, conditional));
  }
}
