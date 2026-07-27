#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

function die(message) { console.error(message); process.exit(1); }
function uniq(xs) { return [...new Set(xs)]; }
function inc(map, key, n = 1) { map[key] = (map[key] || 0) + n; }

try {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) die('usage: ua-arch-analyze.js <input.json> <output.json>');
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = input.fileNodes || [];
  const imports = input.importEdges || [];
  const allEdges = input.allEdges || [];
  const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));

  const split = p => (p || '').split('/').filter(Boolean);
  const paths = nodes.map(n => split(n.filePath));
  let common = [];
  if (paths.length) {
    for (let i = 0; ; i++) {
      const v = paths[0][i];
      if (v === undefined || !paths.every(parts => parts[i] === v)) break;
      common.push(v);
    }
  }
  // A common filename is not a directory prefix.
  if (common.length && paths.every(parts => parts.length === common.length)) common.pop();
  const commonPrefix = common.length ? common.join('/') + '/' : '';
  const directoryGroups = {};
  const groupById = {};
  for (const n of nodes) {
    const parts = split(n.filePath).slice(common.length);
    let group;
    if (parts.length > 1) group = parts[0];
    else if (!common.length && split(n.filePath).length > 1) group = split(n.filePath)[0];
    else group = 'root';
    (directoryGroups[group] ||= []).push(n.id);
    groupById[n.id] = group;
  }

  const nodeTypeGroups = {};
  for (const n of nodes) (nodeTypeGroups[n.type] ||= []).push(n.id);
  const fileFanIn = Object.fromEntries(nodes.map(n => [n.id, 0]));
  const fileFanOut = Object.fromEntries(nodes.map(n => [n.id, 0]));
  const importAdjacency = Object.fromEntries(nodes.map(n => [n.id, []]));
  const interCounts = {};
  const involving = {}, internal = {}, groupImportsFrom = {}, groupImportedBy = {};
  for (const g of Object.keys(directoryGroups)) {
    involving[g] = 0; internal[g] = 0; groupImportsFrom[g] = new Set(); groupImportedBy[g] = new Set();
  }
  for (const e of imports) {
    if (!nodeById[e.source] || !nodeById[e.target]) continue;
    fileFanOut[e.source]++; fileFanIn[e.target]++;
    importAdjacency[e.source].push(e.target);
    const a = groupById[e.source], b = groupById[e.target];
    involving[a]++;
    if (a !== b) involving[b]++;
    else internal[a]++;
    if (a !== b) {
      inc(interCounts, `${a}\u0000${b}`);
      groupImportsFrom[a].add(b); groupImportedBy[b].add(a);
    }
  }
  const interGroupImports = Object.entries(interCounts).map(([k,count]) => {
    const [from,to] = k.split('\u0000'); return {from,to,count};
  }).sort((a,b) => b.count-a.count || a.from.localeCompare(b.from));
  const intraGroupDensity = {};
  for (const g of Object.keys(directoryGroups)) intraGroupDensity[g] = {
    internalEdges: internal[g], totalEdges: involving[g], density: involving[g] ? internal[g] / involving[g] : 0,
    importsFrom: [...groupImportsFrom[g]].sort(), importedBy: [...groupImportedBy[g]].sort()
  };

  const dirPatterns = {
    routes:'api',api:'api',controllers:'api',controller:'api',endpoints:'api',handlers:'api',serializers:'api',routers:'api',blueprints:'api',
    services:'service',core:'service',lib:'service',domain:'service',logic:'service',internal:'service',signals:'service',composables:'service',mailers:'service',jobs:'service',channels:'service',
    models:'data',db:'data',data:'data',persistence:'data',repository:'data',entities:'data',entity:'data',migrations:'data',sql:'data',database:'data',schema:'data',
    components:'ui',views:'ui',pages:'ui',ui:'ui',layouts:'ui',screens:'ui',middleware:'middleware',plugins:'middleware',interceptors:'middleware',guards:'middleware',
    utils:'utility',helpers:'utility',common:'utility',shared:'utility',tools:'utility',pkg:'utility',config:'config',configs:'config',constants:'config',env:'config',settings:'config',management:'config',commands:'config',
    '__tests__':'test',test:'test',tests:'test',spec:'test',specs:'test',types:'types',interfaces:'types',schemas:'types',contracts:'types',dtos:'types',dto:'types',request:'types',response:'types',
    hooks:'hooks',store:'state',state:'state',reducers:'state',actions:'state',slices:'state',assets:'assets',static:'assets',public:'assets',cmd:'entry',bin:'entry',
    docs:'documentation',documentation:'documentation',wiki:'documentation',deploy:'infrastructure',deployment:'infrastructure',infra:'infrastructure',infrastructure:'infrastructure',k8s:'infrastructure',kubernetes:'infrastructure',helm:'infrastructure',charts:'infrastructure',terraform:'infrastructure',tf:'infrastructure',docker:'infrastructure',
    '.github':'ci-cd','.gitlab':'ci-cd','.circleci':'ci-cd'
  };
  const filePattern = p => {
    const base = path.posix.basename(p), low = p.toLowerCase();
    if (/\.(test|spec)\./.test(base) || /^test_.*\.py$/.test(base) || /_test\.go$/.test(base) || /test(s)?\.(java|cs)$/.test(base) || /_spec\.rb$/.test(base)) return 'test';
    if (/\.d\.ts$/.test(base)) return 'types';
    if (['index.ts','index.js','__init__.py','manage.py','main.rs','lib.rs','Application.java','Program.cs','config.ru'].includes(base)) return 'entry';
    if (['wsgi.py','asgi.py','Cargo.toml','go.mod','Gemfile','pom.xml','build.gradle','composer.json'].includes(base)) return 'config';
    if (/^dockerfile/i.test(base) || /^docker-compose\./.test(base) || /\.tf(vars)?$/.test(base) || base === 'Makefile') return 'infrastructure';
    if (low.startsWith('.github/workflows/') || base === '.gitlab-ci.yml' || base === 'Jenkinsfile') return 'ci-cd';
    if (/\.sql$/.test(base)) return 'data';
    if (/\.(graphql|gql|proto)$/.test(base)) return 'types';
    if (/\.(md|rst)$/.test(base)) return 'documentation';
    return null;
  };
  const patternMatches = {};
  for (const [g, ids] of Object.entries(directoryGroups)) {
    patternMatches[g] = dirPatterns[g.toLowerCase()] || null;
    if (!patternMatches[g]) {
      const pats = ids.map(id => filePattern(nodeById[id].filePath)).filter(Boolean);
      if (pats.length && pats.every(x => x === pats[0])) patternMatches[g] = pats[0];
    }
  }
  const filePatternMatches = Object.fromEntries(nodes.map(n => [n.id, filePattern(n.filePath)]).filter(([,v]) => v));

  const cross = {};
  const nonCodeConnections = [];
  for (const e of allEdges) {
    const s=nodeById[e.source], t=nodeById[e.target]; if (!s || !t) continue;
    inc(cross, `${s.type}\u0000${t.type}\u0000${e.type}`);
    if (s.type !== 'file' || t.type !== 'file') nonCodeConnections.push({source:e.source,target:e.target,type:e.type});
  }
  const crossCategoryEdges = Object.entries(cross).map(([k,count]) => {
    const [fromType,toType,edgeType]=k.split('\u0000'); return {fromType,toType,edgeType,count};
  });

  const nodePaths = nodes.map(n => n.filePath);
  const infraFiles = nodePaths.filter(p => /(^|\/)(Dockerfile[^/]*|docker-compose\.[^/]+|Jenkinsfile)$|(^|\/)\.github\/workflows\/|\.gitlab-ci\.yml$|\.(tf|tfvars)$|(^|\/)(k8s|kubernetes|helm|charts)\//i.test(p));
  const deploymentTopology = {
    hasDockerfile: infraFiles.some(p => /(^|\/)Dockerfile/i.test(p)),
    hasCompose: infraFiles.some(p => /docker-compose\./i.test(p)),
    hasK8s: infraFiles.some(p => /(^|\/)(k8s|kubernetes|helm|charts)\//i.test(p)),
    hasTerraform: infraFiles.some(p => /\.(tf|tfvars)$/.test(p)),
    hasCI: infraFiles.some(p => /\.github\/workflows\/|\.gitlab-ci\.yml$|Jenkinsfile$/.test(p)),
    infraFiles
  };
  const dataPipeline = {
    schemaFiles: nodes.filter(n => ['schema','table'].includes(n.type) || /\.(sql|graphql|gql|proto|prisma)$/.test(n.filePath)).map(n=>n.filePath),
    migrationFiles: nodePaths.filter(p => /(^|\/)migrations?\//i.test(p)),
    dataModelFiles: nodePaths.filter(p => /(^|\/)(models?|entities|scene)\//i.test(p)),
    apiHandlerFiles: nodes.filter(n => n.type === 'endpoint' || /(^|\/)(routes?|controllers?|handlers?|api)\//i.test(n.filePath)).map(n=>n.filePath)
  };
  const docs = nodes.filter(n => n.type === 'document' || /\.(md|rst)$/.test(n.filePath));
  const groupHasDocs = {};
  for (const g of Object.keys(directoryGroups)) groupHasDocs[g] = docs.some(d => groupById[d.id] === g || (d.summary || '').toLowerCase().includes(g.toLowerCase()));
  const groupsWithDocs = Object.values(groupHasDocs).filter(Boolean).length;
  const docCoverage = {groupsWithDocs,totalGroups:Object.keys(directoryGroups).length,coverageRatio:Object.keys(directoryGroups).length ? groupsWithDocs/Object.keys(directoryGroups).length : 0,undocumentedGroups:Object.keys(groupHasDocs).filter(g=>!groupHasDocs[g])};
  const pairs = new Set(interGroupImports.map(x => [x.from,x.to].sort().join('\u0000')));
  const dependencyDirection = [];
  for (const key of pairs) {
    const [a,b]=key.split('\u0000'), ab=interCounts[`${a}\u0000${b}`]||0, ba=interCounts[`${b}\u0000${a}`]||0;
    if (ab>ba) dependencyDirection.push({dependent:a,dependsOn:b,count:ab,reverseCount:ba});
    else if (ba>ab) dependencyDirection.push({dependent:b,dependsOn:a,count:ba,reverseCount:ab});
    else dependencyDirection.push({dependent:a,dependsOn:b,count:ab,reverseCount:ba,tied:true});
  }
  const result = {scriptCompleted:true,commonPrefix,directoryGroups,nodeTypeGroups,importAdjacency,crossCategoryEdges,nonCodeConnections,interGroupImports,intraGroupDensity,patternMatches,filePatternMatches,deploymentTopology,dataPipeline,docCoverage,dependencyDirection,fileStats:{totalFileNodes:nodes.length,filesPerGroup:Object.fromEntries(Object.entries(directoryGroups).map(([g,ids])=>[g,ids.length])),nodeTypeCounts:Object.fromEntries(Object.entries(nodeTypeGroups).map(([t,ids])=>[t,ids.length]))},fileFanIn,fileFanOut};
  fs.writeFileSync(outputPath, JSON.stringify(result,null,2)+'\n');
} catch (err) { die(err.stack || String(err)); }
