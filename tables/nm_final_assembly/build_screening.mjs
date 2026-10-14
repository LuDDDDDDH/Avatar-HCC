import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile } from '../workbook.mjs';
import { inputPath, outputPath, outputDirectory, label } from '../inputs.mjs';

const authoredOutput = outputPath('authoredChanges');
const planOutput = outputPath('assemblyPlan');
await fs.mkdir(outputDirectory('qa'), { recursive: true });
const src = JSON.parse(await fs.readFile(inputPath('sourceData'), 'utf8'));
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath('screeningBase')));
const sheetName = 'S1_screening_details';
const sh = wb.worksheets.add(sheetName);
sh.showGridLines = false;
sh.tabColor = '#193F5F';
const body = sh.getRange('A1:F41');
body.format.font = {name:'Arial',size:11,color:'#243746'};
body.format.verticalAlignment = 'center';
body.format.wrapText = true;
const widths = [125,350,95,105,115,610];
widths.forEach((width,i)=>sh.getRangeByIndexes(0,i,41,1).format.columnWidthPx=width);
sh.mergeCells('A1:F2');
sh.getRange('A1').values = [[src.title]];
sh.getRange('A1:F2').format = {fill:'#193F5F',font:{name:'Arial',size:15,bold:true,color:'#FFFFFF'},wrapText:true,verticalAlignment:'center'};
sh.getRange('A1:F2').format.rowHeight = 22;
sh.getRange('A3:F3').format.rowHeight = 10;
sh.getRange('A4:F35').values = src.typed_rows;
sh.getRange('A4:F4').format = {fill:'#3C7CAF',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center'};
sh.getRange('A4:F4').format.rowHeight = 48;
sh.getRange('C5:E35').setNumberFormat('#,##0');
sh.getRange('C5:E35').format.horizontalAlignment = 'center';
sh.getRange('A5:B35').format.horizontalAlignment = 'left';
sh.getRange('F5:F35').format.horizontalAlignment = 'left';
for(let i=1;i<src.typed_rows.length;i++) {
  const r=i+4, values=src.typed_rows[i], range=sh.getRange(`A${r}:F${r}`);
  const section=values[2]===null && values[3]===null && values[4]===null;
  if(section) {
    range.format.fill='#E8F1F6';
    range.format.font.bold=true;
    range.format.font.color='#193F5F';
    range.format.rowHeight=34;
    sh.mergeCells(`B${r}:F${r}`);
  } else {
    range.format.fill=(i%2===0?'#FFFFFF':'#F6F8FA');
    range.format.autofitRows();
    const len = Math.max(Math.ceil(String(values[1]).length/48), Math.ceil(String(values[5]).length/88));
    range.format.rowHeight=Math.max(42, len*15+14);
  }
}
sh.getRange('A36:F36').format.rowHeight=12;
src.notes.forEach((note,i)=>{
  const r=37+i;
  sh.mergeCells(`A${r}:F${r}`);
  sh.getRange(`A${r}`).values=[[note]];
  sh.getRange(`A${r}:F${r}`).format={font:{name:'Arial',size:10,color:'#506879'},wrapText:true,verticalAlignment:'center'};
  sh.getRange(`A${r}:F${r}`).format.rowHeight=Math.max(32,Math.ceil(note.length/200)*14+10);
});
sh.freezePanes.freezeRows(4);
const contents=src.metadata.Contents.map(r=>[...r]);
contents[1][1]=contents[1][1]+'; '+sheetName;
contents[1][2]='Centre-level source counts, final analytical cohorts and detailed screening criteria with stage-specific counts';
const c=wb.worksheets.getItem('Contents');
c.getRange('A1:C12').values=contents;
c.getRange('A2:C2').format.autofitRows();
const dictionary=src.metadata.Data_dictionary.map(r=>[...r]);
dictionary.splice(2,0,[sheetName,'S1','Centre / modality branch × cohort-construction stage','Eligibility, exclusions, pre-assay suitability and assay failures with original counts',label('screeningSource'),'Aggregate counts; parallel modality assessments must not be added together']);
const dict=wb.worksheets.getItem('Data_dictionary');
dict.getRange('A38:F38').copyFrom(dict.getRange('A37:F37'),'all');
dict.getRange('A1:F38').values=dictionary;
dict.getRange('A1:F38').format.autofitRows();
const notes=src.metadata.Notes.map(r=>[...r]);
notes[2][1]=label('screeningScopeNote');
const n=wb.worksheets.getItem('Notes');
n.getRange('A1:B13').values=notes;
n.getRange('A3:B3').format.autofitRows();
wb.recalculate();
const authored=await SpreadsheetFile.exportXlsx(wb);
await authored.save(authoredOutput);
const order=[...src.base_order];
order.splice(order.indexOf('Table_S1_Cohort_composition')+1,0,sheetName);
const plan={sheetName,order,metadata:{Contents:{values:contents,sourceRows:Array.from({length:12},(_,i)=>i+1)},Data_dictionary:{values:dictionary,sourceRows:[1,2,2,...Array.from({length:35},(_,i)=>i+3)]},Notes:{values:notes,sourceRows:Array.from({length:13},(_,i)=>i+1)}}};
await fs.writeFile(planOutput,JSON.stringify(plan,null,2));

