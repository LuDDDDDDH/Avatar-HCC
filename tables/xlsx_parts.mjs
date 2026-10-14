import path from 'node:path';
import JSZip from 'jszip';
import { DOMParser, XMLSerializer } from '@xmldom/xmldom';

const MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main';
const REL = 'http://schemas.openxmlformats.org/package/2006/relationships';
const RID = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships';
const TYPES = 'http://schemas.openxmlformats.org/package/2006/content-types';
const XMLNS = 'http://www.w3.org/2000/xmlns/';
const conditionalSources = new WeakMap();
const conditionalSnapshot = rule => JSON.stringify(Object.fromEntries(Object.entries(rule).filter(([key]) => !['dxfId', 'ref'].includes(key)).sort(([a], [b]) => a.localeCompare(b))));
const parseXML = text => new DOMParser({ onError: (level, message) => { if (level === 'error' || level === 'fatalError') throw new Error(message); } }).parseFromString(text.replace(/^\uFEFF/, ''), 'application/xml');
const serialise = document => new XMLSerializer().serializeToString(document);
const nodes = (document, namespace, local) => Array.from(document.getElementsByTagNameNS(namespace, local));
const relsName = part => path.posix.join(path.posix.dirname(part), '_rels', `${path.posix.basename(part)}.rels`);
const resolvePart = (part, target) => {
  const result = path.posix.normalize(target.startsWith('/') ? target.slice(1) : path.posix.join(path.posix.dirname(part), target));
  if (result.startsWith('../') || result === '..' || result.includes('\\')) throw new Error('Invalid package relationship target');
  return result;
};
const zipText = async (zip, name) => {
  const file = zip.file(name);
  if (!file) throw new Error(`Missing package part: ${name}`);
  return file.async('string');
};

export async function normalizeSpreadsheetNamespaces(buffer) {
  const zip = await JSZip.loadAsync(buffer);
  let changed = false;
  for (const name of Object.keys(zip.files)) {
    if (!name.startsWith('xl/') || !name.endsWith('.xml') || zip.files[name].dir) continue;
    const text = await zipText(zip, name);
    if (!text.includes(MAIN)) continue;
    const document = parseXML(text);
    let omitDrawing = false;
    if (name.startsWith('xl/worksheets/')) {
      const relationshipName = relsName(name);
      const relationships = zip.file(relationshipName) ? parseXML(await zipText(zip, relationshipName)) : null;
      const lookup = new Map(relationships ? nodes(relationships, REL, 'Relationship').map(node => [node.getAttribute('Id'), node]) : []);
      for (const drawing of nodes(document, MAIN, 'drawing')) {
        const relationship = lookup.get(drawing.getAttributeNS(RID, 'id'));
        if (!relationship || relationship.getAttribute('TargetMode') === 'External') continue;
        const graph = await relationshipGraph(zip, resolvePart(name, relationship.getAttribute('Target')));
        if (![...graph].some(part => /(?:^|\/)charts\/chart[^/]*\.xml$/.test(part))) continue;
        drawing.parentNode.removeChild(drawing); omitDrawing = true;
      }
    }
    if (!omitDrawing && !nodes(document, MAIN, '*').some(node => node.prefix)) continue;
    const normal = document.implementation.createDocument(null, null, null);
    const copy = node => {
      if (node.nodeType !== 1) return normal.importNode(node, true);
      const element = normal.createElementNS(node.namespaceURI, node.namespaceURI === MAIN ? node.localName : node.tagName);
      for (const attribute of Array.from(node.attributes)) element.setAttributeNS(attribute.namespaceURI, attribute.name, attribute.value);
      if (node === document.documentElement && node.namespaceURI === MAIN) element.setAttributeNS(XMLNS, 'xmlns', MAIN);
      for (const child of Array.from(node.childNodes)) element.appendChild(copy(child));
      return element;
    };
    for (const child of Array.from(document.childNodes)) normal.appendChild(copy(child));
    zip.file(name, serialise(normal));
    changed = true;
  }
  return changed ? zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }) : buffer;
}

async function worksheetParts(zip) {
  const workbook = parseXML(await zipText(zip, 'xl/workbook.xml'));
  const relationships = parseXML(await zipText(zip, 'xl/_rels/workbook.xml.rels'));
  const map = new Map(nodes(relationships, REL, 'Relationship').map(node => [node.getAttribute('Id'), node.getAttribute('Target')]));
  return new Map(nodes(workbook, MAIN, 'sheet').map(node => {
    const target = map.get(node.getAttributeNS(RID, 'id'));
    if (!target) throw new Error('Worksheet relationship is missing');
    return [node.getAttribute('name'), resolvePart('xl/workbook.xml', target)];
  }));
}

export async function restoreConditionalRules(buffer, workbook) {
  const zip = await JSZip.loadAsync(buffer);
  const styles = parseXML(await zipText(zip, 'xl/styles.xml'));
  const formats = nodes(styles, MAIN, 'dxfs')[0];
  const differentials = formats ? Array.from(formats.childNodes).filter(node => node.nodeType === 1 && node.localName === 'dxf') : [];
  for (const [name, part] of await worksheetParts(zip)) {
    const sheet = workbook.getWorksheet(name);
    if (!sheet) throw new Error('Conditional-formatting worksheet is missing');
    const document = parseXML(await zipText(zip, part));
    for (const group of nodes(document, MAIN, 'conditionalFormatting')) {
      const reference = group.getAttribute('sqref');
      const targetGroups = sheet.conditionalFormattings.filter(item => item.ref === reference);
      for (const node of nodes(group, MAIN, 'cfRule')) {
        const priority = Number(node.getAttribute('priority'));
        const matches = targetGroups.flatMap(item => item.rules).filter(rule => rule.priority === priority);
        if (matches.length !== 1) throw new Error('Cannot identify a conditional-formatting rule');
        const rule = matches[0];
        if (node.hasAttribute('text')) {
          rule.text = node.getAttribute('text');
          if (node.hasAttribute('operator')) rule.operator = node.getAttribute('operator');
          rule.formulae = nodes(node, MAIN, 'formula').map(formula => formula.textContent);
        }
        let dxf = null;
        if (node.hasAttribute('dxfId')) {
          const item = differentials[Number(node.getAttribute('dxfId'))];
          if (!item) throw new Error('Conditional-formatting differential style is missing');
          dxf = serialise(item);
        }
        conditionalSources.set(rule, { xml: serialise(node), dxf, snapshot: conditionalSnapshot(rule) });
      }
    }
  }
}

export async function preserveConditionalRules(buffer, workbook) {
  const entries = workbook.worksheets.flatMap(sheet => sheet.conditionalFormattings.flatMap(group => group.rules
    .filter(rule => conditionalSources.has(rule) || typeof rule.text === 'string')
    .map(rule => ({ name: sheet.name, reference: group.ref, rule }))));
  if (!entries.length) return buffer;
  const zip = await JSZip.loadAsync(buffer);
  const parts = await worksheetParts(zip);
  const documents = new Map();
  const references = new Map();
  const recordReference = index => references.set(index, (references.get(index) ?? 0) + 1);
  for (const [name, part] of parts) {
    const document = parseXML(await zipText(zip, part));
    documents.set(name, document);
    for (const node of nodes(document, MAIN, 'cfRule')) if (node.hasAttribute('dxfId')) recordReference(Number(node.getAttribute('dxfId')));
  }
  for (const part of Object.keys(zip.files).filter(name => name.startsWith('xl/tables/') && name.endsWith('.xml'))) {
    const document = parseXML(await zipText(zip, part));
    for (const node of Array.from(document.getElementsByTagName('*'))) for (const attribute of Array.from(node.attributes)) {
      if (/dxfId$/i.test(attribute.localName)) recordReference(Number(attribute.value));
    }
  }
  const pending = [];
  for (const name of new Set(entries.map(entry => entry.name))) {
    const part = parts.get(name);
    if (!part) throw new Error('Exported conditional-formatting worksheet is missing');
    const document = documents.get(name);
    for (const { reference, rule } of entries.filter(entry => entry.name === name)) {
      const groups = nodes(document, MAIN, 'conditionalFormatting').filter(node => node.getAttribute('sqref') === reference);
      const matches = groups.flatMap(group => nodes(group, MAIN, 'cfRule')).filter(node => Number(node.getAttribute('priority')) === rule.priority);
      if (matches.length !== 1) throw new Error('Cannot identify an exported conditional-formatting rule');
      const node = matches[0];
      const source = conditionalSources.get(rule);
      if (source?.snapshot === conditionalSnapshot(rule)) pending.push({ source, document, node });
      else if (typeof rule.text === 'string') {
        node.setAttribute('text', rule.text);
        if (rule.operator) node.setAttribute('operator', rule.operator);
      }
    }
  }
  const styles = parseXML(await zipText(zip, 'xl/styles.xml'));
  let formats = nodes(styles, MAIN, 'dxfs')[0];
  let styleChanged = false;
  const claims = new Map();
  for (const item of pending) if (item.source.dxf) {
    const index = item.node.hasAttribute('dxfId') ? Number(item.node.getAttribute('dxfId')) : null;
    if (!claims.has(index)) claims.set(index, []);
    claims.get(index).push(item);
  }
  const added = new Map();
  const differentials = () => formats ? Array.from(formats.childNodes).filter(node => node.nodeType === 1 && node.localName === 'dxf') : [];
  for (const [index, items] of claims) {
    const originals = new Set(items.map(item => item.source.dxf));
    const current = differentials();
    if (index !== null && originals.size === 1 && references.get(index) === items.length && current[index]) {
      formats.replaceChild(styles.importNode(parseXML(items[0].source.dxf).documentElement, true), current[index]);
      for (const item of items) item.dxfId = index;
      styleChanged = true;
    } else for (const item of items) {
      if (!formats) {
        formats = styles.createElementNS(MAIN, 'dxfs');
        const later = Array.from(styles.documentElement.childNodes).find(node => node.nodeType === 1 && ['tableStyles', 'colors', 'extLst'].includes(node.localName));
        styles.documentElement.insertBefore(formats, later ?? null);
      }
      if (!added.has(item.source.dxf)) {
        const next = differentials().length;
        formats.appendChild(styles.importNode(parseXML(item.source.dxf).documentElement, true));
        added.set(item.source.dxf, next);
      }
      item.dxfId = added.get(item.source.dxf);
      styleChanged = true;
    }
  }
  if (styleChanged) {
    formats.setAttribute('count', String(differentials().length));
    zip.file('xl/styles.xml', serialise(styles));
  }
  for (const { source, document, node, dxfId } of pending) {
    const restored = document.importNode(parseXML(source.xml).documentElement, true);
    restored.setAttribute('priority', node.getAttribute('priority'));
    if (dxfId !== undefined) restored.setAttribute('dxfId', String(dxfId));
    node.parentNode.replaceChild(restored, node);
  }
  for (const name of new Set(entries.map(entry => entry.name))) zip.file(parts.get(name), serialise(documents.get(name)));
  return zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
}

async function relationshipGraph(zip, part, collected = new Set()) {
  if (collected.has(part)) return collected;
  if (!zip.file(part)) throw new Error(`Missing drawing relationship part: ${part}`);
  collected.add(part);
  const relationshipPart = relsName(part);
  if (zip.file(relationshipPart)) {
    collected.add(relationshipPart);
    const document = parseXML(await zipText(zip, relationshipPart));
    for (const relationship of nodes(document, REL, 'Relationship')) {
      if (relationship.getAttribute('TargetMode') === 'External') continue;
      await relationshipGraph(zip, resolvePart(part, relationship.getAttribute('Target')), collected);
    }
  }
  return collected;
}

function applyContentTypes(source, output, copied, mapPart) {
  const defaults = new Map(nodes(output, TYPES, 'Default').map(node => [node.getAttribute('Extension'), node]));
  const overrides = new Map(nodes(output, TYPES, 'Override').map(node => [node.getAttribute('PartName'), node]));
  const sourceOverrides = new Set(nodes(source, TYPES, 'Override').map(node => node.getAttribute('PartName').replace(/^\//, '')));
  for (const node of nodes(source, TYPES, 'Default')) {
    const extension = node.getAttribute('Extension');
    if (![...copied].some(part => part.endsWith(`.${extension}`) && !sourceOverrides.has(part))) continue;
    const existing = defaults.get(extension);
    if (existing && existing.getAttribute('ContentType') !== node.getAttribute('ContentType')) throw new Error('Conflicting drawing content type');
    if (!existing) {
      const item = output.importNode(node, true);
      output.documentElement.appendChild(item); defaults.set(extension, item);
    }
  }
  for (const node of nodes(source, TYPES, 'Override')) {
    const original = node.getAttribute('PartName').replace(/^\//, '');
    if (!copied.has(original)) continue;
    const mapped = `/${mapPart(original)}`;
    const existing = overrides.get(mapped);
    if (existing && existing.getAttribute('ContentType') !== node.getAttribute('ContentType')) throw new Error('Conflicting drawing override');
    if (!existing) {
      const item = output.importNode(node, true);
      item.setAttribute('PartName', mapped);
      output.documentElement.appendChild(item); overrides.set(mapped, item);
    }
  }
}

export async function preserveNativeCharts(sourceBuffer, outputBuffer) {
  if (!sourceBuffer) return outputBuffer;
  const source = await JSZip.loadAsync(sourceBuffer);
  const nativeCharts = Object.keys(source.files).filter(name => /(?:^|\/)charts\/chart[^/]*\.xml$/.test(name));
  if (!nativeCharts.length) return outputBuffer;
  const output = await JSZip.loadAsync(outputBuffer);
  const before = await worksheetParts(source), after = await worksheetParts(output);
  const copied = new Set();
  const mapPart = part => part;
  const rootLinks = [];
  for (const [name, worksheetPart] of before) {
    const worksheet = parseXML(await zipText(source, worksheetPart));
    const drawings = nodes(worksheet, MAIN, 'drawing');
    if (!drawings.length) continue;
    const relPart = relsName(worksheetPart);
    const relationships = parseXML(await zipText(source, relPart));
    const lookup = new Map(nodes(relationships, REL, 'Relationship').map(node => [node.getAttribute('Id'), node]));
    for (const drawing of drawings) {
      const relationship = lookup.get(drawing.getAttributeNS(RID, 'id'));
      if (!relationship || relationship.getAttribute('TargetMode') === 'External') throw new Error('Native drawing relationship is missing or external');
      const drawingPart = resolvePart(worksheetPart, relationship.getAttribute('Target'));
      const graph = await relationshipGraph(source, drawingPart);
      if (!nativeCharts.some(part => graph.has(part))) continue;
      if (!after.has(name)) throw new Error('Cannot preserve a chart from a removed worksheet');
      for (const part of graph) copied.add(part);
      rootLinks.push({ name, drawingPart, type: relationship.getAttribute('Type') });
    }
  }
  if (nativeCharts.some(part => !copied.has(part))) throw new Error('Native charts are not connected to preserved worksheet drawings');
  for (const part of copied) {
    const destination = mapPart(part);
    let bytes = await source.file(part).async('nodebuffer');
    if (part.endsWith('.rels')) {
      const document = parseXML(bytes.toString('utf8'));
      let changed = false;
      for (const relationship of nodes(document, REL, 'Relationship')) {
        if (relationship.getAttribute('TargetMode') === 'External') continue;
        const target = relationship.getAttribute('Target');
        if (!target.startsWith('/')) continue;
        const original = target.slice(1);
        if (!copied.has(original)) throw new Error('Unresolved absolute drawing relationship');
        const mappedTarget = `/${mapPart(original)}`;
        if (mappedTarget !== target) { relationship.setAttribute('Target', mappedTarget); changed = true; }
      }
      if (changed) bytes = Buffer.from(serialise(document));
    }
    if (output.file(destination) && !(await output.file(destination).async('nodebuffer')).equals(bytes)) throw new Error('Drawing part destination collision');
    output.file(destination, bytes);
  }
  for (const name of new Set(rootLinks.map(link => link.name))) {
    const worksheetPart = after.get(name);
    const worksheet = parseXML(await zipText(output, worksheetPart));
    const relPart = relsName(worksheetPart);
    const relationships = output.file(relPart) ? parseXML(await zipText(output, relPart)) : parseXML(`<Relationships xmlns="${REL}"/>`);
    for (const node of nodes(worksheet, MAIN, 'drawing')) node.parentNode.removeChild(node);
    for (const node of nodes(relationships, REL, 'Relationship')) if (node.getAttribute('Type').endsWith('/drawing')) node.parentNode.removeChild(node);
    const existing = new Set(nodes(relationships, REL, 'Relationship').map(node => node.getAttribute('Id')));
    let index = 1;
    for (const link of rootLinks.filter(item => item.name === name)) {
      let id;
      do { id = `rIdAvatarDrawing${index++}`; } while (existing.has(id));
      existing.add(id);
      const relationship = relationships.createElementNS(REL, 'Relationship');
      relationship.setAttribute('Id', id); relationship.setAttribute('Type', link.type);
      relationship.setAttribute('Target', path.posix.relative(path.posix.dirname(worksheetPart), mapPart(link.drawingPart)));
      relationships.documentElement.appendChild(relationship);
      const drawing = worksheet.createElementNS(MAIN, 'drawing'); drawing.setAttributeNS(RID, 'r:id', id);
      const later = Array.from(worksheet.documentElement.childNodes).find(node => node.nodeType === 1 && ['legacyDrawing', 'legacyDrawingHF', 'picture', 'oleObjects', 'controls', 'webPublishItems', 'tableParts', 'extLst'].includes(node.localName));
      worksheet.documentElement.insertBefore(drawing, later ?? null);
    }
    output.file(worksheetPart, serialise(worksheet)); output.file(relPart, serialise(relationships));
  }
  const sourceTypes = parseXML(await zipText(source, '[Content_Types].xml'));
  const outputTypes = parseXML(await zipText(output, '[Content_Types].xml'));
  applyContentTypes(sourceTypes, outputTypes, copied, mapPart);
  output.file('[Content_Types].xml', serialise(outputTypes));
  return output.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
}
