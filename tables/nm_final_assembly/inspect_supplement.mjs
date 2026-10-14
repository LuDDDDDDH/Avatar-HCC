import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, SpreadsheetFile } from '../workbook.mjs';
import { inputPath, outputPath, outputDirectory } from '../inputs.mjs';

const plan=JSON.parse(await fs.readFile(inputPath('assemblyPlan'),'utf8'));
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath('nmOutput')));
const summary=(await wb.inspect({kind:'sheet',include:'id,name',maxChars:20000})).ndjson;
const names=summary.split('\n').filter(Boolean).map(x=>JSON.parse(x)).filter(x=>x.kind==='sheet').map(x=>x.name);
if(JSON.stringify(names)!==JSON.stringify(plan.order)) throw new Error('Wrong worksheet order');
wb.recalculate();
const errors=(await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:2000})).ndjson;
await fs.mkdir(outputDirectory('qa'), {recursive:true});
await fs.writeFile(outputPath('inspectionErrors'),errors);

for(const [sheetName,range,name] of [
 ['Contents','A1:C12','contents'],
 ['S1_screening_details','A1:F11','detail_1'],
 ['S1_screening_details','A12:F20','detail_2'],
 ['S1_screening_details','A21:F29','detail_3'],
 ['S1_screening_details','A30:F41','detail_4'],
 ['Data_dictionary','A1:F8','dictionary'],
 ['Notes','A1:B5','notes']
]) {
 if(process.argv.includes('--detail-only') && !name.startsWith('detail_')) continue;
 const blob=await wb.render({sheetName,range,scale:1.5,format:'png',fitRows:true});
 await fs.writeFile(path.join(outputDirectory('qa'),name+'.png'),new Uint8Array(await blob.arrayBuffer()));

}
