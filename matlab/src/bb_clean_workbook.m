function bb_clean_workbook(file)
% Remove only empty default Sheet1/2/3 OOXML worksheets; no Excel automation.
folder=tempname;mkdir(folder);cleanup=onCleanup(@()rmdir(folder,'s')); %#ok<NASGU>
unzip(file,folder);book=fullfile(folder,'xl','workbook.xml');rels=fullfile(folder,'xl','_rels','workbook.xml.rels');
b=xmlread(book);r=xmlread(rels);types=fullfile(folder,'[Content_Types].xml');t=xmlread(types);
sheets=b.getElementsByTagName('sheet');changed=false;
for j=sheets.getLength-1:-1:0
 node=sheets.item(j);name=char(node.getAttribute('name'));
 if ~any(strcmp(name,{'Sheet1','Sheet2','Sheet3'})),continue;end
 id=char(node.getAttribute('r:id'));links=r.getElementsByTagName('Relationship');
 for k=links.getLength-1:-1:0
  link=links.item(k);if ~strcmp(char(link.getAttribute('Id')),id),continue;end
  target=char(link.getAttribute('Target'));
  if target(1)=='/',relative=target(2:end);else,relative=['xl/' target];end
  path=fullfile(folder,strrep(relative,'/',filesep));sheet=xmlread(path);
  % Conservatively preserve cells, formulas, inline strings and drawings.
  if sheet.getElementsByTagName('c').getLength>0||sheet.getElementsByTagName('drawing').getLength>0,continue;end
  node.getParentNode.removeChild(node);link.getParentNode.removeChild(link);
  entries=t.getElementsByTagName('Override');
  for h=entries.getLength-1:-1:0
   entry=entries.item(h);
   if strcmp(char(entry.getAttribute('PartName')),['/' relative]),entry.getParentNode.removeChild(entry);end
  end
  delete(path);changed=true;
 end
end
if changed
 xmlwrite(book,b);xmlwrite(rels,r);xmlwrite(types,t);
 archive=[tempname '.zip'];archiveCleanup=onCleanup(@()localDelete(archive)); %#ok<NASGU>
 files=dir(folder);names={files(~ismember({files.name},{'.','..'})).name};
 zip(archive,names,folder);movefile(archive,file,'f');
end
end
function localDelete(file)
if exist(file,'file'),delete(file);end
end
